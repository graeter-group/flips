# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PDB data loader."""
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
from tqdm import tqdm

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pathlib import Path
import os

from gafl.analysis import utils as au
from gafl.analysis.metrics import calc_mdtraj_metrics
from gafl.data.pdb_dataloader import PdbDataset as PdbDatasetGAFL
from gafl.data.pdb_dataloader import PdbDataModule as PdbDataModuleGAFL
from gafl.data import utils as du
import warnings
import ast


PICKLE_EXTENSIONS = ['.pkl', '.pickle', '.pck', '.db', '.pck']


class PdbDataModule(PdbDataModuleGAFL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage: str):
        if self.dataset_cfg.calc_dssp:
            pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)
            # Check if csc has columns helix_percent, strand_percent, coil_percent
            if not ('helix_percent' in pdb_csv.columns and 'strand_percent' in pdb_csv.columns):
                logging.info("Calculating DSSP values for dataset")
                #Iterate over csv column 'processed_path'
                helix_pct = []
                strand_pct = []
                coil_pct = []

                for path in tqdm(pdb_csv['processed_path']):
                    try:
                        path_extension = Path(path).suffix

                        processed_feats = du.get_processed_feats(path, path_extension)
                        
                    except Exception as e:
                            raise ValueError(f'Error in processing {path}') from e
                    
                    atom_pos = processed_feats['atom_positions']
                    os.makedirs(os.path.join('tmp'), exist_ok=True)
                    au.write_prot_to_pdb(atom_pos, os.path.join('tmp', 'dssp_sample.pdb'), overwrite=True, no_indexing=True)

                    #Calculate dssp
                    dssp = calc_mdtraj_metrics(os.path.join('tmp', 'dssp_sample.pdb'))
                    helix_pct.append(dssp['helix_percent'])
                    strand_pct.append(dssp['strand_percent'])
                    coil_pct.append(dssp['coil_percent'])
            
                #Append dssp values to csv
                pdb_csv['helix_percent'] = helix_pct
                pdb_csv['strand_percent'] = strand_pct
                pdb_csv['coil_percent'] = coil_pct
                pdb_csv.to_csv(self.dataset_cfg.csv_path, index=False)

        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False,
        )
        self._valid_sample_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False,
            sample_dataset=True,
        )

class PdbDataset(PdbDatasetGAFL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overwrite process_csv_row
    def _process_csv_row(self, processed_file_path):
        try:
            processed_feats = du.get_processed_feats(processed_file_path, extra_feats=self.dataset_cfg.extra_features if hasattr(self.dataset_cfg, 'extra_features') else None)
            modeled_idx = processed_feats['modeled_idx']
            if len(modeled_idx) == 0:
                raise ValueError(f'No modeled residues found in {processed_file_path}')

            # Filter out residues that are not modeled.
            min_idx = np.min(modeled_idx)
            max_idx = np.max(modeled_idx)

            processed_feats = tree.map_structure(
                    lambda x: x[min_idx:(max_idx+1)], processed_feats)
  
            # Run through OpenFold data transforms.
            chain_feats = {
                'aatype': torch.tensor(processed_feats['aatype']).long(),
                'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
                'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
            }
            chain_feats = data_transforms.atom37_to_frames(chain_feats)
            rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
            rotmats_1 = rigids_1.get_rots().get_rot_mats()
            trans_1 = rigids_1.get_trans()
            res_idx = processed_feats['residue_index']
            res_idx = res_idx - np.min(res_idx) + 1

            feat_dict = {
                'aatype': chain_feats['aatype'],
                'res_idx': torch.tensor(res_idx),
                'rotmats_1': rotmats_1,
                'trans_1': trans_1,
                'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
                'atom_positions': chain_feats['all_atom_positions'],
            }

            # load extra features:
            if hasattr(self.dataset_cfg, 'extra_features'):
                for extra_feature in self.dataset_cfg.extra_features:
                    # checks:
                    if extra_feature in feat_dict.keys():
                        raise ValueError(f'Feature {extra_feature} already exists in feat_dict')
                    if extra_feature not in processed_feats.keys():
                        raise ValueError(f'Feature {extra_feature} not found in processed_feats')
                    if processed_feats[extra_feature] is None:
                        raise ValueError(f'Feature {extra_feature} is None in processed_feats')
                    
                    feat_dict[extra_feature] = torch.tensor(processed_feats[extra_feature]).float()
                    # TODO: from mask_flexibility to here
                    if feat_dict[extra_feature].shape[0] != feat_dict['trans_1'].shape[0]:
                        print(len(feat_dict[extra_feature].shape))
                        if len(feat_dict[extra_feature].shape) != 1 and feat_dict[extra_feature].shape[-1] == 1:
                            feat_dict[extra_feature].squeeze(-1)
                            warnings.warn(f"Feature {extra_feature}: Squeezing last dimension")
                        raise ValueError(f"Feature {extra_feature}: shape mismatch: {feat_dict[extra_feature].shape[0]} != {feat_dict['trans_1'].shape[0]}")

        except Exception as e:
            raise ValueError(f'Error in processing {processed_file_path}') from e
        
        return feat_dict
        
    def __getitem__(self, idx, conf_idx:int=None):
            # Sample data example.
            example_idx = idx
            if isinstance(example_idx, list):
                example_idx = example_idx[0]

            csv_row = self.csv.iloc[example_idx]
            processed_file_path = csv_row['processed_path']
            chain_feats = self._process_csv_row(processed_file_path)
            chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx

            if self.dataset_cfg.use_res_idx:
                # If there are inconsistencies in the different break definitions, consecutive residue indices are used
                if (not csv_row['consistent_breaks']):
                    chain_feats['res_idx'] = torch.arange(chain_feats['res_mask'].shape[0], dtype=torch.long)
            else:
                chain_feats['res_idx'] = torch.arange(chain_feats['res_mask'].shape[0], dtype=torch.long)

            if self.dataset_cfg.label_breaks:
                break_mask = torch.zeros(chain_feats['res_mask'].shape[0], dtype=torch.float32)
                break_idc = csv_row['merged_idx']
                # if there are no breaks, break_idc is empty, so we can leave the mask as it is
                if len(break_idc) > 0:
                    if isinstance(break_idc, str):
                        break_idc = ast.literal_eval(break_idc)
                    break_idc = np.array(break_idc, dtype=np.int32)
                    break_idc = np.append(break_idc, break_idc + 1)
                    #Create tensor of length N with 1 at break positions
                    break_mask[break_idc] = 1.0

                chain_feats['breaks'] = break_mask

            if not self.is_training:
                chain_feats['target_helix_percent'] = self.helix_percent
                chain_feats['target_strand_percent'] = self.strand_percent

            if 'local_flex' in chain_feats.keys():
                if len(chain_feats['local_flex'].shape) != 1 and chain_feats['local_flex'].shape[-1] == 1:
                    warnings.warn(f"Local flex had additional dimension: Squeezing last dimension")
                    chain_feats['local_flex'] = chain_feats['local_flex'].squeeze(-1)
            return chain_feats