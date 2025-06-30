# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
import subprocess
from typing import Optional
from biotite.sequence.io import fasta
import pandas as pd

from gafl import experiment_utils as eu
from gafl.data import utils as du
from gafl.analysis import metrics
from tqdm import tqdm

log = eu.get_pylogger(__name__)

def run_self_consistency_batch(
        pmpnn_dir: str,
        sequences_per_sample: int,
        folding_model,
        length_dir: str,
        sample_ids: np.ndarray,
        motif_mask: Optional[np.ndarray]=None,
        calc_non_coil_rmsd: bool = True,
        max_res_per_esm_batch: int = 12000,
        weights:str='default',
    ):
    """Run self-consistency on design proteins against reference protein.
    
    Args:
        decoy_pdb_dir: directory where designed protein files are stored.
        reference_pdb_path: path to reference protein file
        motif_mask: Optional mask of which residues are the motif.

    Returns:
        Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
        Writes ESMFold outputs to decoy_pdb_dir/esmf
        Writes results in decoy_pdb_dir/sc_results.csv
    """

    # clear gpu memory:
    torch.cuda.empty_cache()

    with torch.no_grad():
        fasta_headers = []
        fasta_seqs = []
        for sample_id in tqdm(sample_ids, desc="ProteinMPNN for samples"):
            sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
            decoy_pdb_dir = os.path.join(sample_dir, 'self_consistency')
            reference_pdb_path = os.path.join(sample_dir, 'sample.pdb')

            # Sample random seed
            seed = np.random.randint(0, 999)
            seed = str(int(seed))
            
            # Run ProteinMPNN
            output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
            process = subprocess.Popen([
                'python',
                f'{pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
                f'--input_path={decoy_pdb_dir}',
                f'--output_path={output_path}',
            ])
            _ = process.wait()
            num_tries = 0
            ret = -1
            
            pmpnn_args = [
                'python',
                f'{pmpnn_dir}protein_mpnn_run.py',
                '--out_folder', decoy_pdb_dir,
                '--jsonl_path', output_path,
                '--num_seq_per_target', str(sequences_per_sample),
                '--sampling_temp', '0.1',
                '--seed', seed,
                '--batch_size', '1',
            ]
            
            if weights == 'soluble':
                log.info("Using soluble ProteinMPNN weights")
                pmpnn_args.append('--use_soluble_model')
            pmpnn_args.extend(['--model_name', 'v_48_020'])

            # make the call silent, also for errors:
            command = ' '.join(pmpnn_args) + ' >/dev/null 2>&1'
            os.system(command)

            mpnn_fasta_path = os.path.join(
                decoy_pdb_dir,
                'seqs',
                os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
            )

            if not os.path.exists(mpnn_fasta_path):
                raise FileNotFoundError(f"ProteinMPNN failed to generate fasta file: {mpnn_fasta_path}")

            fasta_file = fasta.FastaFile.read(mpnn_fasta_path)
            fasta_headers.extend(list(fasta_file.keys())[1:])
            fasta_seqs.extend(list(fasta_file.values())[1:])


        log.info(f"Run ESM on {len(fasta_seqs)} sequences")        
        # get the number of residues for each sequence:
        seq_lens = [len(seq) for seq in fasta_seqs]
        # use max_num_res for splitting the list of fasta_seqs into batches:
        batches = []
        batch = []
        batch_len = 0
        for i, seq_len in enumerate(seq_lens):
            if batch_len + seq_len > max_res_per_esm_batch:
                batches.append(batch)
                batch = []
                batch_len = 0
            batch.append(fasta_seqs[i])
            batch_len += seq_len
        if len(batch) > 0:
            batches.append(batch)
        log.info(f"Run ESM on batches of size {[len(b) for b in batches]} with {len(batches)} batches")

        all_esm_pdbs = []
        for batch in tqdm(batches, "Running ESMFold"):
            # clear gpu memory:
            torch.cuda.empty_cache()
            all_esm_output = folding_model.infer(batch)
            all_esm_pdbs.extend(folding_model.output_to_pdb(all_esm_output))

        # calculate metrics
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if calc_non_coil_rmsd:
            mpnn_results['non_coil_rmsd'] = []
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []

        # for i in tqdm(range(len(fasta_seqs)), desc="Calc metrics and save"):
        for i in range(len(fasta_seqs)):
            sample_id = sample_ids[i//sequences_per_sample]
            header = fasta_headers[i]
            sequence = fasta_seqs[i]
            esm_pdb = all_esm_pdbs[i]
            sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
            decoy_pdb_dir = os.path.join(sample_dir, 'self_consistency')
            reference_pdb_path = os.path.join(sample_dir, 'sample.pdb')

            esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
            os.makedirs(esmf_dir, exist_ok=True)

            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i%sequences_per_sample}.pdb')
            with open(esmf_sample_path, "w") as f:
                f.write(esm_pdb)

            sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path, calc_dssp=calc_non_coil_rmsd)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])

            if calc_non_coil_rmsd:
                # calculate the rmsd if coils in the refolded structure are ignored:
                non_coil_idxs = np.where(esmf_feats['dssp'] != 'C')[0]
                if len(non_coil_idxs) == 0:
                    non_coil_rmsd = np.nan
                else:
                    non_coil_rmsd = metrics.calc_aligned_rmsd(
                        sample_feats['bb_positions'][non_coil_idxs],
                        esmf_feats['bb_positions'][non_coil_idxs])

            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)

            if calc_non_coil_rmsd:
                mpnn_results['non_coil_rmsd'].append(non_coil_rmsd)
                
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(sequence)

            if (i+1) % sequences_per_sample == 0:
                if i > 0:
                    # Save results to CSV
                    csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
                    mpnn_results = pd.DataFrame(mpnn_results)
                    mpnn_results.to_csv(csv_path)
                mpnn_results = {
                    'tm_score': [],
                    'sample_path': [],
                    'header': [],
                    'sequence': [],
                    'rmsd': [],
                }
                if calc_non_coil_rmsd:
                    mpnn_results['non_coil_rmsd'] = []