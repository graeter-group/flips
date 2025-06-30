import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Optional
import shutil
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

class MetricsMultiLen:
    def __init__(self, results_folder: str,
                 compute_designability:bool=False,
                 num_top_samples: Optional[int] = 1,
                 prediction_method: Optional[str] = 'esmf', 
                 compute_novelty:bool=False,
                 dataset_foldseek: Optional[str] = None,):
        
        self.results_folder = results_folder
        self.metadata_results = None
        self.top_samples = num_top_samples
        self.prediction_method = prediction_method
        self.compute_designability = compute_designability
        self.compute_novelty = compute_novelty

        if self.compute_novelty:
            if self.dataset_foldseek is None:
                raise ValueError("dataset_foldseek must be provided to compute novelty scores.")
            self.dataset_foldseek = dataset_foldseek

        self.best_samples_folder = os.path.join(self.results_folder, 'best_samples')
        if self.compute_designability:
            self.best_refolded_folder = os.path.join(self.results_folder, 'best_refolded_samples')
        self.best_flex_profiles_folder = os.path.join(self.results_folder, 'best_flex_profiles')

    def get_flex_scores(self, sample_path: str, w_corr: int = 1, w_mae: int = 2):
        """Load flexibility files and compute correlation and mae for scoring samples"""
        flex_target = np.loadtxt(os.path.join(sample_path, 'local_flex_target.txt'))
        pred_flex_target = np.loadtxt(os.path.join(sample_path, 'local_flex_predicted.txt'))
        flex_correlation = np.corrcoef(pred_flex_target, flex_target)[0, 1]
        mae = np.mean(np.abs(pred_flex_target - flex_target))
        score = w_corr * flex_correlation - w_mae * mae
        return flex_correlation, mae, score

    def get_designability(self, sample_path: str, rmsd_cutoff: float = 2.0):
        """Load self-consistency metrics and determine if the sample is designable. In the paper we considered a sample designable if the minumum scRMSD is less than 2.0."""
        sc_metrics = os.path.join(sample_path, 'self_consistency', 'sc_results.csv')
        sc_metrics_df = pd.read_csv(sc_metrics)
        vals = sc_metrics_df['rmsd'].to_numpy()
        sample_val = np.min(vals)
        designable = sample_val <= rmsd_cutoff
        return designable, sample_val

    def get_flex_metrics_per_pen(self):
        """
        Aggregates results per sampled length.
        Returns a DataFrame of top_k samples for each length, sorted by score and rmsd.
        """
        len_folders = [i for i in os.listdir(self.results_folder) if os.path.isdir(os.path.join(self.results_folder, i))]
        metadata_results = {'length': [], 'sample': [], 'designable': [], 'rmsd':[], 'flex_corr': [], 'flex_mae': [], 'score': []}

        for sampled_length in tqdm(len_folders, desc='Aggregating flexibility results'):
            metadata_length = {'sample': [], 'designable': [], 'rmsd':[], 'flex_corr': [], 'flex_mae':[], 'score': []}

            top_flex = os.path.join(self.results_folder, sampled_length, 'top_flex')
            if not os.path.exists(top_flex):
                print(f'{top_flex} does not exist.')
                continue

            len_folder_top_samples = os.path.join(top_flex, sampled_length)
            top_samples = [i for i in os.listdir(len_folder_top_samples) if os.path.isdir(os.path.join(len_folder_top_samples, i))]

            for sample in top_samples:
                sample_path = os.path.join(len_folder_top_samples, sample)
                try:
                    flex_correlation, mae, score = self.get_flex_scores(sample_path)
                    if self.compute_designability:
                        designable, rmsd = self.get_designability(sample_path)
                    else:
                        designable, rmsd = False, np.nan  # If designability is not computed, set to False and NaN
                except Exception as e:
                    print(f"Error in {sample_path}: {e}, skipping the sample")
                    continue

                metadata_length['sample'].append(sample)
                metadata_length['designable'].append(designable)
                metadata_length['rmsd'].append(rmsd)
                metadata_length['flex_corr'].append(flex_correlation)
                metadata_length['flex_mae'].append(mae)
                metadata_length['score'].append(score)

            # Collect designable samples
            df = pd.DataFrame(metadata_length)
            df['length'] = sampled_length  # add length info for each sample
            
            # Sort by score descending, then rmsd ascending
            df_sorted = df.sort_values(by=['score', 'rmsd'], ascending=[False, True])

            # Take top-k samples
            top_k_df = df_sorted.head(self.top_samples)

            # Append to metadata_results
            for _, row in top_k_df.iterrows():
                metadata_results['length'].append(row['length'])
                metadata_results['sample'].append(row['sample'])
                metadata_results['designable'].append(row['designable'])
                metadata_results['rmsd'].append(row['rmsd'])
                metadata_results['flex_corr'].append(row['flex_corr'])
                metadata_results['flex_mae'].append(row['flex_mae'])
                metadata_results['score'].append(row['score'])

        self.metadata_results = pd.DataFrame(metadata_results)
        self.metadata_results.sort_values(by='score', ascending=False, inplace=True)
        self.metadata_results.to_csv(os.path.join(self.results_folder, 'flex_results.csv'), index=False)
        return self.metadata_results

    def aggregate_best_samples(self, best_samples_folder: str):
        """Copy the best top_k samples to the output folder."""
        if self.metadata_results is None:
            raise ValueError("Metadata results are not available. Run extract_score_per_len_multilen first.")

        os.makedirs(best_samples_folder, exist_ok=True)
        best_samples = self.metadata_results.sort_values(by=['score'], ascending=[False, True])
        
        for _, row in best_samples.iterrows():
            length_sample = row['length']
            sample = row['sample']
            sample_loc = os.path.join(self.results_folder, f'{length_sample}', 'top_flex', f'{length_sample}', f'{sample}', 'sample.pdb')
            shutil.copy(sample_loc, os.path.join(best_samples_folder, f'{length_sample}_{sample}.pdb'))

    def aggregate_best_refolded_samples(self, best_refolded_folder: str):
        """
        Copy the best top_k refolded samples to the output folder
        """
        if self.metadata_results is None:
            raise ValueError("Metadata results are not available. Run extract_score_per_len_multilen first.")

        os.makedirs(best_refolded_folder, exist_ok=True)
        best_samples = self.metadata_results.sort_values(by=['score', 'rmsd'], ascending=[False, True])

        for _, row in best_samples.iterrows():
            length_sample = row['length']
            sample = row['sample']
            sc_metrics = os.path.join(self.results_folder, f'{length_sample}', 'top_flex', f'{length_sample}', f'{sample}', 'self_consistency', 'sc_results.csv')
            sc_metrics_df = pd.read_csv(sc_metrics)
            vals = sc_metrics_df['rmsd'].to_numpy()
            idx_select = np.argmin(vals)
            sample_numb = sc_metrics_df['sample_path'][idx_select].split('/')[-1]
            sample_loc_abs = os.path.join(self.results_folder, f'{length_sample}', 'top_flex', f'{length_sample}', f'{sample}', 'self_consistency', self.prediction_method, sample_numb)
            refolded_number = sample_numb.split('_')[-1].split('.')[0]
            shutil.copy(sample_loc_abs, os.path.join(best_refolded_folder, f'{length_sample}_{sample}_refolded_{refolded_number}.pdb'))

    def aggregate_best_flex_profiles(self, best_flex_profiles_folder: str):
        """Copy all flexibility profiles to the output folder."""
        if self.metadata_results is None:
            raise ValueError("Metadata results are not available. Run extract_score_per_len_multilen first.")

        os.makedirs(best_flex_profiles_folder, exist_ok=True)
        for _, row in self.metadata_results.iterrows():
            length_sample = row['length']
            sample = row['sample']
            local_flex_png = os.path.join(self.results_folder, f'{length_sample}', 'top_flex', f'{length_sample}', f'{sample}', 'local_flex.png')
            shutil.copy(local_flex_png, os.path.join(best_flex_profiles_folder, f'{length_sample}_{sample}_flex.png'))

    def compute_novelty_folder(self):
        """
        Computes TM score between the input pdb (e.g. one wants to use as a condition) and the closest hit in the reference database (.pdb formatted) with FoldSeek.
        Args:
            - dataset_foldseek:str - path to the folder with reference pdbs (.pdb format) for FoldSeek
            - input_pdb_folder:str - path to the folder with the input pdb file(s)
        """
        if not os.path.exists(os.path.join(self.best_samples_folder, 'aln.csv')):
            print(f'{self.best_samples_folder}/aln.csv does not exist. Running FoldSeek easy-search...')
            os.system(f"foldseek easy-search {self.best_samples_folder} {self.dataset_foldseek} {self.best_samples_folder}/aln.csv {self.best_samples_folder}/tmpFolder --format-mode 4 --format-output query,target,alntmscore,ttmscore,evalue --alignment-type 1")
        novelty_df = pd.read_csv(f"{self.best_samples_folder}/aln.csv", sep='\t')
        novelty_df = novelty_df.sort_values(by='evalue', ascending=False)
        self.novelty_df = novelty_df
    
    def get_novelty_all(self):
        
        metadata_copy = self.metadata_results.copy()
        self.novelty_df.set_index('query', inplace=True)
        for index in tqdm(self.novelty_df.index.unique()):
            split_groups = index.split('_')
            length = split_groups[0]+'_'+split_groups[1]
            sample = split_groups[2]+'_'+split_groups[3]

            vals = self.novelty_df.loc[index]['evalue']
            names = self.novelty_df.loc[index]['target']

            if isinstance(vals, pd.Series):
                vals = vals.values
            else:	
                vals = [vals]
            max_idx = np.argmax(vals)
            novelty = vals[max_idx]

            metadata_copy.loc[
                (metadata_copy['length'] == length) &
                (metadata_copy['sample'] == sample),
                'novelty'
            ] = novelty

        print(f"Novelty scores computed for {len(metadata_copy)} samples. Saving results...")
        metadata_copy.to_csv(os.path.join(self.results_folder, 'flex_results.csv'), index=False)

    def compute_metrics(self):
        self.get_flex_metrics_per_pen()

        self.aggregate_best_samples(best_samples_folder=self.best_samples_folder)
        if self.compute_designability:
            self.aggregate_best_refolded_samples(best_refolded_folder=self.best_refolded_folder)
        self.aggregate_best_flex_profiles(best_flex_profiles_folder=self.best_flex_profiles_folder)
        
        if self.compute_novelty:
            self.compute_novelty_folder()
            self.metadata_results = self.get_novelty_all()

def get_args():
    parser = argparse.ArgumentParser(description="Compute flexibility and designability metrics for generated proteins.")

    parser.add_argument('--results_folder', type=str, required=True,
                        help='Path to the folder containing results produced by FliPS.')
    parser.add_argument('--compute_designability', action='store_true',
                        help='Whether to compute designability scores based on self-consistency metrics.')
    parser.add_argument('--num_top_samples', type=int, default=10,
                        help='Number of top flexibility-ranked samples to keep per sampled length.')
    parser.add_argument('--prediction_method', type=str, default='esmf',
                        help="Prediction method used for structure prediction.")
    parser.add_argument('--compute_novelty', action='store_true',
                        help='NOTE: requires FoldSeek to be installed. Whether to compute FoldSeek novelty scores.')
    parser.add_argument('--dataset_foldseek', type=str, default=None,
                        help='Path to reference dataset folder (.pdb format) used for novelty calculation via FoldSeek.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    metrics = MetricsMultiLen(
        results_folder=args.results_folder,
        compute_designability=args.compute_designability,
        num_top_samples=args.num_top_samples,
        prediction_method=args.prediction_method,
        compute_novelty=args.compute_novelty, # Set to True if you want to compute novelty scores
        dataset_foldseek=args.dataset_foldseek # Path to the dataset for FoldSeek novelty calculation
    )
    metrics.compute_metrics()