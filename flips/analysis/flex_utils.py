# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import biotite.structure.io.pdb as pdb_io
from gafl.analysis import utils as au

from flips.analysis.summary import ExperimentMetrics

def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aatype = None,
        local_flex:np.ndarray = None,
        predicted_local_flex:np.ndarray = None,
        write_traj:bool=False
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        local_flex: [N] The target local flexibility condition of each residue given by the user.
        predicted_local_flex: [N] predicted local flexibility of each residue of the generated sample.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')
    local_flex_path = os.path.join(output_dir, 'local_flex_target.txt')
    predicted_local_flex_path = os.path.join(output_dir, 'local_flex_predicted.txt')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    if write_traj:
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aatype,
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aatype
        )

    if local_flex is not None:
        np.savetxt(local_flex_path, local_flex)

    if predicted_local_flex is not None:
        np.savetxt(predicted_local_flex_path, predicted_local_flex)

    if predicted_local_flex is not None and local_flex is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(local_flex, label='Target')
        ax.plot(predicted_local_flex, label='Predicted')
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Local RMSF')
        ax.set_title('Flexibility of sample')
        ax.legend()
        fig.savefig(os.path.join(output_dir, 'local_flex.png'))
        plt.close(fig)
        
    if write_traj:
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }
    else:
        return {
            'sample_path': sample_path,
        }

def self_consistency_evaluation(output_dir, ckpt_path, designability_mode='scrmsd'):
    
    data = ExperimentMetrics(path=output_dir, force_update=True, designability_mode=designability_mode)
    data.calc_metrics()
    data.calc_novelty() #Novelty calculation will need some additional setup (foldseek and datasets)
    data.calc_mdtraj_metrics()
    data.save()
    summary = data.summary()

    # store a summary of the results in a json file and a text file in the output directory:
    summary_path = Path(output_dir) / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    summary_csv_path = Path(output_dir) / "summary.csv"
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.to_csv(summary_csv_path)

    summary_txt_path = Path(output_dir) / "summary.txt"
    with open(summary_txt_path, "w") as f:
        f.write(str(data))

    
    # store the results json file to the checkpoint directory:
    summary_path = Path(ckpt_path).parent / "results" / (Path(ckpt_path).stem + ".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # load all results json files in the checkpoint directory and merge them into a single json file, sorted by the designability entry:
    all_results_ = {}
    for path in Path(ckpt_path).parent.glob("results/*.json"):
        if path.name == "all_results.json":
            continue
        with open(path, "r") as f:
            all_results_[path.stem] = json.load(f)
    
    # sort the dictionary:
    all_results = {}
    sorted_keys = sorted(all_results_.keys(), key=lambda x: all_results_[x]["designability"][0], reverse=True)
    for k in sorted_keys:
        all_results[k] = all_results_[k]

    all_results_path = summary_path.parent / "all_results.json"

    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # now transform this to a dataframe and save it in a pretty format:
    for k in all_results:
        for kk in list(all_results[k].keys()):
            if "scrmsd" in kk:
                all_results[k].pop(kk)
                continue
            if isinstance(all_results[k][kk], list):
                all_results[k][kk] = str(round(all_results[k][kk][0],2)) + " ± " + str(round(all_results[k][kk][1],2))

    # now replace the keys by their first 3 and last 9 letters:
    formatted_keys = {k[-9:] if len(k) > 9 else k: v for k, v in all_results.items()}

    df = pd.DataFrame.from_dict(formatted_keys, orient='index')  # Important: use orient='index' to treat keys as row labels

    df_string = df.to_string()

    txt_path = summary_path.parent / "summary.txt"
    with open(txt_path, "w") as f:
        f.write(df_string)

    return df

def get_structure_backbone(pdb_loc:str, allowed_atoms=['CA'], b_factors=False):
    with open(pdb_loc, "r") as file_handle:
        pdb_file = pdb_io.PDBFile.read(file_handle)
    protein = pdb_io.get_structure(pdb_file, model=1, extra_fields=['b_factor'])
    # returning only those residues that are not heteroatoms and do not have insertion codes
    protein = protein[(protein.hetero == False) & (protein.ins_code == '')]
    protein = protein[np.isin(protein.atom_name, allowed_atoms)]
    if b_factors:
        return protein
    else:
        protein.b_factor = np.zeros(protein.shape[0])
        return protein

def make_profile_from_bfacs(loc_pdb:str, safe_path:str):
    test_denovo_profile = get_structure_backbone(loc_pdb, b_factors=True)
    with open(safe_path, 'w') as f:
        np.savetxt(f, test_denovo_profile.b_factor, fmt='%.3f')