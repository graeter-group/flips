# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script to analyze the inference performance of trained protein models
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import biotite.structure.io as bsio
from pypdb import get_info
import Bio.PDB
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import gaussian_kde
import json
import shutil

from gafl.analysis import metrics
from gafl.analysis import utils as au
import gafl.data.utils as du


def compare_metrics(models, metrics, pred_name=None, include_num_length=False, sort=None):
    data = {}
    index = {
        "model": []
    }
    
    valid_models = []
    valid_models_names = []

    for i, m in enumerate(models):
        if not m.is_empty:
            valid_models.append(m)
            valid_models_names.append(pred_name[i])
    
    models = valid_models
    pred_name = valid_models_names

    for j, v in enumerate(metrics):
        data[v] = []
        for i, m in enumerate(models):

            index["model"].append(pred_name[i])
            
            val, _ = m.get_mean(v)
            data[v].append(val)

    if include_num_length:
        data['num_length'] = []
        for i, m in enumerate(models):
            data['num_length'].append(len(np.unique(m.get('length').to_numpy())))
                                          
    df = pd.DataFrame.from_dict(data).set_index(pd.Index(pred_name))

    if sort is None:
        #Sort df by index
        df = df.sort_index()
    else:
        # Sort df by column specified in "sort"
        df = df.sort_values(by=sort, ascending=False)

    df = df.style.format("{:.4f}")
    display(df)

def plot_metrics(model_groups, epochs, metrics, names, dataset=None, baseline=None):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(figsize=(18, 7), ncols=3, nrows=2)
    
    data = {}
    err = {}

    lines = []  # List to store the Line2D instances for the legend
    labels = []  # List to store the labels for the legend

    for i, metric in enumerate(metrics):

        for j, group in enumerate(model_groups):
            data[metric] = []
            err[metric] = []

            for model in group:
                val, error = model.get_mean(metric)
                data[metric].append(val)
                err[metric].append(error)
                #Plot with solid line style
            line = axs[i // 3, i % 3].errorbar(epochs[j], data[metric], yerr=err[metric], label=names[j], linestyle='-', marker='o')
            if i == 0:  # Only add the lines/labels once
                lines.append(line)
                labels.append(names[j])

        if dataset is not None:
            val, error = dataset.get_mean(metric)
            if val is not None:
                line = axs[i // 3, i % 3].axhline(y=val, color='grey', linestyle='-', label='SCOPe')
                if i == 0:  # Only add the lines/labels once
                    lines.append(line)
                    labels.append('SCOPe')

        if baseline is not None:
            val, error = baseline.get_mean(metric)
            if val is not None:
                line = axs[i // 3, i % 3].axhline(y=val, color='grey', linestyle='--', label='Baseline')
                if i == 0:  # Only add the lines/labels once
                    lines.append(line)
                    labels.append('Baseline')

        axs[i // 3, i % 3].set_title(metric)

     # Create a single legend for all subplots
    fig.legend(lines, labels, loc='right')

def tradeoff_plot(model_groups, metrics, names, dataset=None, baseline=None):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(figsize=(8, 8), ncols=1, nrows=1)


    for j, group in enumerate(model_groups):
        metric1 = []
        metric2 = []
        for model in group:
            metric1.append(model.get_mean(metrics[0])[0])
            metric2.append(model.get_mean(metrics[1])[0])
            #Plot with solid line style
        axs.plot(metric1, metric2, label=names[j], linestyle='-', marker='o')

    if dataset is not None:
        val1, _ = dataset.get_mean(metrics[0])
        val2, _ = dataset.get_mean(metrics[1])
        if val1 is not None and val2 is not None:
            axs.scatter(val1, val2, color='black', label='SCOPe')

    if baseline is not None:
        val1, _ = baseline.get_mean(metrics[0])
        val2, _ = baseline.get_mean(metrics[1])
        if val1 is not None and val2 is not None:
            axs.scatter(val1, val2, color='red', label='Baseline')

    axs.set_xlabel(metrics[0])
    axs.set_ylabel(metrics[1])
    axs.set_title(f'{metrics[1]} vs {metrics[0]}')
    axs.legend()
    fig.show()

class Metrics:

    def __init__(self, path, name, save_path=None, designability_mode='scrmsd'):
        self.path = path
        self.name = name
        self.data = None
        self.mask = None

        if save_path is None:
            self.save_path = self.path
        else:
            self.save_path = save_path

        self.infostr = None
        self.infodict = None
        self.novelty_hits = []
        self.novelty_TM_by_sample = []
        assert designability_mode in ['scrmsd', 'nc_scrmsd'], "Designability mode must be 'scrmsd' or 'nc_scrmsd'"
        self.designability_mode = designability_mode

        self.sample_dirs = []
    
    def save(self):
        if self.data is not None:
            self.data.to_csv(os.path.join(self.save_path, 'metrics.csv'), index=False)
    
    def get(self, key):
        if self.data is None:
            print("Metrics not found. Please run calc_metrics() first.")
            return None
        if key not in self.data.columns:
            print(f"Key {key} not found in metrics.")
            return None
        if self.mask is None:
            return self.data[key]
        else:
            return self.data[key][self.mask]
        
    def set_mask(self, mask):
        """
        Mask to apply to columns of self.data when retrieved via .get()
        """
        self.mask = mask

    def mask_length(self, min_l, max_l):
        if self.data is None:
            print("Metrics not found. Please run calc_metrics() first.")
            return
        self.mask = np.logical_and(self.data['length'] >= min_l, self.data['length'] <= max_l)
        return self

    def mask_intervall(self, key, min_val, max_val):
        if self.data is None:
            print("Metrics not found. Please run calc_metrics() first.")
            return
        self.mask = np.logical_and(self.data[key] >= min_val, self.data[key] <= max_val)
        return self

    def _get_infostring(self, mask=None):
        '''
        In this function, the mean and standard deviation of the metrics are calculated and returned as a string/dictionary.
        '''

        if self.infostr is not None or self.infodict is not None:
            assert self.infodict is not None and self.infostr is not None, "Internal error: Infostr and infodict must be set together"
            return self.infostr, self.infodict
        
        #Check if metric file exists
        if not os.path.exists(os.path.join(self.path, "metrics.csv")):
            print("Metrics file not found. Please run calc_metrics() first.")
            return

        infostr = ""
        infodict = {}

        if mask is None and self.mask is None:
            mask = np.ones(len(self.data), dtype=bool)
            infostr += f"-------- {self.name} Metrics --------\n"
        elif mask is None and self.mask is not None:
            mask = self.mask
            masked_fraction = (1 - np.mean(mask.astype(int)))*100
            infostr += f"-------- {self.name} Metrics ({masked_fraction:.2f}% masked) --------\n"
        else:
            masked_fraction = (1 - np.mean(mask.astype(int)))*100
            infostr += f"-------- {self.name} Metrics ({masked_fraction:.2f}% masked) --------\n"
        
        metrics = self.data[mask]

        ignore_keys = ['length', 'sample', 'pdb', 'exptl']

        for key in metrics.columns:
            if key in ignore_keys:
                continue
            val, err = self.get_mean(key)
            try:
                infostr += f"{key}: {val:.4f} +/- {err:.4f}\n"
            except:
                try:
                    infostr += f"{key}: {val}\n"
                except:
                    infostr += f"\n"
            infodict[key] = [val, err]

        if 'scrmsd' in metrics.columns:
            val, err = self.get_mean('designability', calc_err=True)            

            try:
                infostr += f"Designability: {val:.4f} +/- {err:.4f}\n"
            except:
                try:
                    infostr += f"Designability: {val}\n"
                except:
                    infostr += f"\n"
            infodict['designability'] = [val, err]
        
        if 'non_coil_scrmsd' in metrics.columns:
            val, err = self.get_mean('nc_designability', calc_err=True)
            try:
                infostr += f"Non-coil Designability: {val:.4f} +/- {err:.4f}\n"
            except:
                try:
                    infostr += f"Non-coil Designability: {val}\n"
                except:
                    infostr += f"\n"

            infodict['nc_designability'] = [val, err]

        infostr += "------------------------------\n"

        self.infostr = infostr
        self.infodict = infodict

        return infostr, infodict

    def __str__(self) -> str:
        return self._get_infostring()[0]

    def summary(self, mask=None):
        infostr, infodict = self._get_infostring(mask)
        print(infostr)
        with open(os.path.join(self.save_path, 'summary.json'), 'w') as f:
            json.dump(infodict, f, indent=4)

        # for each sample, store a sample_metrics.json file containing the scrmsd, nc_scrmsd, length, novelty, helix content, strand content, coil content:
        for i in range(len(self.data)):
            sample_info = {k: float(self.data[k][i]) for k in ['length', 'scrmsd', 'non_coil_scrmsd', 'helix_percent', 'strand_percent', 'coil_percent'] if k in self.data.columns}
            sample_info['novelty_TM'] = str(self.novelty_TM_by_sample[i] if len(self.novelty_TM_by_sample) > 0 else None)
            sample_info['novelty_closest_hit'] = str(self.novelty_hits[i] if len(self.novelty_hits) > 0 else None)

            with open(os.path.join(self.sample_dirs[i], f'sample_metrics.json'), 'w') as f:
                json.dump(sample_info, f, indent=4)

        return infodict
        

    def get_mean(self, key, calc_err=True):
        DESIGNABILITY_KEYS = [('designability', 'scrmsd'), ('nc_designability', 'non_coil_scrmsd')]
        REF_VALUE_LIMIT = 2

        if key in [x[0] for x in DESIGNABILITY_KEYS]:
            ref_value = [x[1] for x in DESIGNABILITY_KEYS if x[0] == key][0]
            scrmsd = self.get(ref_value).to_numpy()
            if len(scrmsd) == 0:
                return None, None
            
            # scrmsd = scrmsd[~np.isnan(scrmsd)]
            if np.any(np.isnan(scrmsd)):
                raise ValueError(f"NaN values found in {ref_value} column!")

            val = np.mean((scrmsd < REF_VALUE_LIMIT).astype(int))
            
            #Estimate error on designability
            n = len(scrmsd)
            bs_samples = []
            if calc_err:
                for i in range(100):
                    idxs = np.random.choice(n, n)
                    bs_samples.append(np.mean((scrmsd[idxs] < 2).astype(int)))
                err = np.std(bs_samples)
            else:
                err = None
            return val, err

        elif key not in self.data.columns:
            return None, None

        # elif key == 'novelty':
        #     tm = self.get('novelty').to_numpy()
        #     tm = tm[~np.isnan(tm)]
        #     return np.mean((tm < 0.7).astype(int)), 0

        else:
            vals = self.get(key).to_numpy()
            if len(vals) == 0:
                return None, None
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return float('nan'), float('nan')
            return np.mean(vals), np.std(vals) / np.sqrt(len(vals))

    def eval_correlation(self, key1, key2, ax=None):
        vals1 = self.get(key1)
        vals2 = self.get(key2)
        mask = np.logical_and(~np.isnan(vals1), ~np.isnan(vals2))
        correlation = np.corrcoef(self.get(key1)[mask], self.get(key2)[mask])[0, 1]

        if ax is not None:
            # ax.scatter(self.get(key1), self.get(key2), s=1, label=f'corr = {correlation:.2f}')
            # Make colored scatter plot with kde
            xy = np.vstack([self.get(key1)[mask], self.get(key2)[mask]])
            z = gaussian_kde(xy)(xy)
            ax.scatter(self.get(key1)[mask], self.get(key2)[mask], c=z, s=10, cmap='viridis', label=f'corr = {correlation:.2f}')

            ax.set_xlabel(key1)
            ax.set_ylabel(key2)

            ax.set_title(f'{self.name} {key1} vs {key2}', fontsize=12)
            ax.legend()

        return correlation

class ExperimentMetrics(Metrics):

    def __init__(self, path, name="Experiment", force_update=False, tmp_path=None, save_path=None, **kwargs):
        super().__init__(path, name, save_path=save_path, **kwargs)

        if tmp_path is None:
            self.tmp_path = self.path
        else:
            self.tmp_path = tmp_path

        if os.path.exists(os.path.join(self.save_path, 'metrics.csv')) and not force_update:
            self.data = pd.read_csv(os.path.join(self.save_path, 'metrics.csv'))
            # Change old naming conventions
            if 'designability' in self.data.columns:
                self.data = self.data.rename(columns={'designability': 'scrmsd'})
            if 'plddt' in self.data.columns:
                self.data = self.data.drop(columns=['plddt'])
        else:
            self.data = None
        
        if self.data is not None:
            self.is_empty = len(self.data) == 0
        else:
            self.is_empty = True
        errs = None

    def calc_metrics(self, calc_diversity=True, prep_novelty=True, calc_nc_diversity=True):
        FLEX = False
        errs = 0
        # Create folder to store designable samples (for novelty calculation)
        if not os.path.exists(os.path.join(self.tmp_path, 'samples')):
            os.makedirs(os.path.join(self.tmp_path, 'samples'))
        os.system(f"rm {os.path.join(self.tmp_path, 'samples')}/*")

        data = {
            'length': [],
            'sample': [],
            'scrmsd': [],
            'non_coil_scrmsd': [],
        }

        if FLEX:
            data['flex_correlation'] = []
            data['flex_rmse'] = []

        if calc_diversity:
            data['diversity'] = []
        if calc_nc_diversity:
            data['nc_diversity'] = []

        folders_ = [file for file in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, file))]
        for length in tqdm(folders_, desc="Calculating designability/diversity"):
            if '_' not in length:
                continue
            prefix, length_val = length.split('_')
            length_val = int(length_val)

            if prefix == 'length':
                diversity_samples = []
                diversity_seqs = []
                diversity_vals = []
                nc_diversity_samples = []
                nc_diversity_seqs = []
                nc_diversity_vals = []
                for sample in os.listdir(os.path.join(self.path, length)):
                    if not os.path.isdir(os.path.join(self.path, length, sample)):
                        continue
                    if '_' not in sample:
                        continue
                    

                    SKIP_ERRS = True
                    errs = 0

                    try:

                        prefix, sample_val = sample.split('_')
                        sample_val = int(sample_val)

                        if prefix == 'sample':
                            self.sample_dirs.append(os.path.join(self.path, length, sample))


                            csv_path = os.path.join(self.path, length, sample, "self_consistency/sc_results.csv")
                            # if non_coil_rmsd is a col, also use this col:
                            if "non_coil_rmsd" in pd.read_csv(csv_path).columns:
                                calc_non_coil_rmsd = True
                            else:
                                calc_non_coil_rmsd = False
                                if FLEX:
                                    raise ValueError("Non-coil rmsd not found in metrics but needed for flexibility prediction.")
                                if calc_nc_diversity:
                                    raise ValueError("Non-coil diversity requested but non_coil_rmsd not found in metrics.")

                            sc_metrics = pd.read_csv(csv_path, usecols=['tm_score', 'rmsd'] + (['non_coil_rmsd'] if calc_non_coil_rmsd else []))
                            scrmsd, sample_idx = self.calc_scrmsd(sc_metrics)

                            # store the pdb file of the best sample at min_scrmsd_sample.pdb:
                            sample_path = os.path.join(self.path, length, sample, 'self_consistency', 'esmf', f'sample_{sample_idx}.pdb')
                            new_sample_path = os.path.join(self.path, length, sample, 'min_scrmsd_sample.pdb')
                            shutil.copyfile(sample_path, new_sample_path)

                            data['scrmsd'].append(scrmsd)
                            data['length'].append(length_val)
                            data['sample'].append(sample_val)

                            # non-coil designability
                            #######################
                            if calc_non_coil_rmsd:
                                non_coil_scrmsd, sample_idx_nc = self.calc_scrmsd(sc_metrics, rmsd_key='non_coil_rmsd')
                                # store the pdb file of the best sample at min_nc_scrmsd_sample.pdb:
                                sample_path = os.path.join(self.path, length, sample, 'self_consistency', 'esmf', f'sample_{sample_idx_nc}.pdb')
                                new_sample_path = os.path.join(self.path, length, sample, 'min_nc_scrmsd_sample.pdb')
                                shutil.copyfile(sample_path, new_sample_path)
                                data['non_coil_scrmsd'].append(non_coil_scrmsd)
                                if non_coil_scrmsd < 2:
                                    if calc_nc_diversity:
                                        sample_feats = du.parse_pdb_feats('sample', os.path.join(self.path, length, sample, "sample.pdb"))
                                        sample_seq = du.aatype_to_seq(sample_feats['aatype'])

                                        nc_diversity_samples.append(sample_feats)
                                        nc_diversity_seqs.append(sample_seq)
                                        nc_diversity_vals.append(1)


                                    if self.designability_mode=='nc_scrmsd':
                                        if prep_novelty:
                                            # Copy sample.pdb to sample folder
                                            pdb_file = os.path.join(self.path, length, sample, "sample.pdb")
                                            new_pdb_file = os.path.join(self.tmp_path, 'samples', f"{length_val}_{sample_val}.pdb")
                                            os.system(f'cp {pdb_file} {new_pdb_file}')
                                            
                                        if FLEX:
                                            pred_flex_path = os.path.join(self.path, length, sample, 'local_flex_predicted.txt')
                                            assert os.path.exists(pred_flex_path), f"Flexibility prediction not found for {length}/{sample}"
                                            if os.path.exists(pred_flex_path):
                                                local_flex = np.loadtxt(pred_flex_path)
                                                target_flex = np.loadtxt(os.path.join(self.path, length, sample, 'local_flex_target.txt'))

                                                flex_correlation = np.corrcoef(local_flex, target_flex)[0, 1]
                                                flex_rmse = np.sqrt(np.mean((local_flex - target_flex)**2))

                                                data['flex_correlation'].append(flex_correlation)
                                                data['flex_rmse'].append(flex_rmse)

                                else:
                                    if calc_nc_diversity:
                                        nc_diversity_vals.append(0)

                                    if FLEX:
                                        data['flex_correlation'].append(None)
                                        data['flex_rmse'].append(None)
                            #######################
                            if scrmsd < 2:
                                if calc_diversity:
                                    sample_feats = du.parse_pdb_feats('sample', os.path.join(self.path, length, sample, "sample.pdb"))
                                    sample_seq = du.aatype_to_seq(sample_feats['aatype'])

                                    diversity_samples.append(sample_feats)
                                    diversity_seqs.append(sample_seq)
                                    diversity_vals.append(1)
                                
                                if self.designability_mode=='scrmsd':
                                    if prep_novelty:
                                        # Copy sample.pdb to sample folder
                                        pdb_file = os.path.join(self.path, length, sample, "sample.pdb")
                                        new_pdb_file = os.path.join(self.tmp_path, 'samples', f"{length_val}_{sample_val}.pdb")
                                        os.system(f'cp {pdb_file} {new_pdb_file}')

                                    if FLEX:
                                        pred_flex_path = os.path.join(self.path, length, sample, 'local_flex_predicted.txt')
                                        assert os.path.exists(pred_flex_path), f"Flexibility prediction not found for {length}/{sample}"
                                        if os.path.exists(pred_flex_path):
                                            local_flex = np.loadtxt(pred_flex_path)
                                            target_flex = np.loadtxt(os.path.join(self.path, length, sample, 'local_flex_target.txt'))

                                            flex_correlation = np.corrcoef(local_flex, target_flex)[0, 1]
                                            flex_rmse = np.sqrt(np.mean((local_flex - target_flex)**2))

                                            data['flex_correlation'].append(flex_correlation)
                                            data['flex_rmse'].append(flex_rmse)

                            else:
                                if calc_diversity:
                                    diversity_vals.append(0)

                    except Exception as e:
                        if not SKIP_ERRS:
                            raise e
                        else:
                            errs += 1
                            print(f"Error processing {length}/{sample}: {e}")

                if calc_diversity:
                    # calculate diversity
                    diversity = self.calc_diversity(diversity_samples, diversity_seqs)
                    diversity_vals = np.array(diversity_vals)
                    diversity_arr =  np.zeros(len(diversity_vals))
                    diversity_arr[diversity_vals == 1] = diversity
                    diversity_arr[diversity_vals == 0] = None
                    data['diversity'].extend(list(diversity_arr))

                if calc_nc_diversity:
                    # calculate non-coil diversity
                    nc_diversity = self.calc_diversity(nc_diversity_samples, nc_diversity_seqs)
                    nc_diversity_vals = np.array(nc_diversity_vals)
                    nc_diversity_arr =  np.zeros(len(nc_diversity_vals))
                    nc_diversity_arr[nc_diversity_vals == 1] = nc_diversity
                    nc_diversity_arr[nc_diversity_vals == 0] = None
                    data['nc_diversity'].extend(list(nc_diversity_arr))
    
        if errs > 0:
            print(f"\nErrors occurred in {errs} samples. Skipping these samples.\n")

        if len(data['non_coil_scrmsd']) == 0:
            data.pop('non_coil_scrmsd')
        elif len(data['non_coil_scrmsd']) != len(data['scrmsd']):
            raise ValueError("Non-coil scrmsd and scrmsd have different lengths.")

        self.data = pd.DataFrame(data)
        self.is_empty = len(self.data) == 0
        self.data = self.data.sort_values(["length", "sample"], ascending=[True,True])
                    
    def calc_scrmsd(self, sc_metrics, rmsd_key='rmsd'):
        """
        Determines if refolded protein is designable and also returns the idx of the ProteinMPNN sequence with the highest designable score
        which is then further used to calculate novelty
        """

        vals = sc_metrics[rmsd_key].to_numpy()

        sample_idx = np.argmin(vals)
        sample_val = np.min(vals)
        return sample_val, sample_idx
    
    def calc_diversity(self, diversity_samples, diversity_seqs):
        """
        Calculate diversity of the refolded proteins
        """

        if len(diversity_samples) < 2:
            return None

        tm_scores = []
        for i in range(len(diversity_samples)):
            for j in range(i+1, len(diversity_samples)):
                _, tm_score = metrics.calc_tm_score(
                    diversity_samples[i]['bb_positions'], diversity_samples[j]['bb_positions'],
                    diversity_seqs[i], diversity_seqs[j])
                tm_scores.append(tm_score)
        
        return np.mean(tm_scores)

    def calc_novelty(self):
        l_min = min(self.data['length'])
        s_min = min(self.data['sample'])
        l_size = max(self.data["length"]) - min(self.data["length"]) + 1
        s_size = max(self.data["sample"]) - min(self.data["sample"]) + 1
        
        novelty = [None] * (len(self.sample_dirs))

        novelty_tms = [float("nan")] * (len(self.sample_dirs))
        novelty_hits = ["None"] * (len(self.sample_dirs))

        # FOLDSEEK_DATA_PATH = '../../data/pdb/pdb'
        FOLDSEEK_DATA_PATH = '/local/user/seutelf/foldseek/pdb' # NOTE: Change this to the path where the PDB files are stored

        SILENT = True
        os.system(f"foldseek easy-search {os.path.join(self.tmp_path, 'samples')}/ {FOLDSEEK_DATA_PATH} {os.path.join(self.tmp_path, 'samples', 'aln.csv')} tmp --format-output query,target,alntmscore --format-mode 4 --alignment-type 1" + (' > /dev/null' if SILENT else ''))
        aln = pd.read_csv(os.path.join(self.tmp_path, 'samples', 'aln.csv'), sep='\t')
        aln.set_index('query', inplace=True)

        for index in tqdm(aln.index.unique(), desc="Calculating novelty"):
            l, s = index.split('_')
            l = int(l)
            s = int(s)

            # sample_idx = (l-l_min)*s_size + (s-s_min) # only works if samples are continuous!!

            sample_id = s
            length = l
            sample_dir = os.path.join(self.path, f"length_{l}", f"sample_{s}")
            assert sample_dir in self.sample_dirs, f"Sample {sample_dir} not found in sample_dirs."
            sample_idx = self.sample_dirs.index(sample_dir)


            vals = aln.loc[index]['alntmscore']
            names = aln.loc[index]['target']
            if isinstance(vals, pd.Series):
                vals = vals.values
            else:
                vals = [vals]

            max_idx = np.argmax(vals)
            novelty[sample_idx] = vals[max_idx]

            novelty_hits[sample_idx] = names[max_idx]
            novelty_tms[sample_idx] = vals[max_idx]

        self.novelty_TM_by_sample = novelty_tms
        self.novelty_hits = novelty_hits

        self.data['novelty'] = novelty

    def calc_mdtraj_metrics(self):
        helix_percent = []
        strand_percent = []
        coil_percent = []
        for l in tqdm(np.unique(self.data['length'].to_numpy()), desc="Calculating MDtraj metrics"):
            for s in np.unique(self.data['sample'].to_numpy()):

                mdtraj_metrics = metrics.calc_mdtraj_metrics(os.path.join(self.path, f"length_{l}", f"sample_{s}", "sample.pdb"))

                helix_percent.append(mdtraj_metrics['helix_percent'])
                strand_percent.append(mdtraj_metrics['strand_percent'])
                coil_percent.append(mdtraj_metrics['coil_percent'])

        self.data['helix_percent'] = helix_percent
        self.data['strand_percent'] = strand_percent
        self.data['coil_percent'] = coil_percent

    def calc_ca_ca_metrics(self):
        parser = Bio.PDB.PDBParser(QUIET=True)

        ca_ca_deviation = []
        ca_ca_valid_percent = []
        num_ca_ca_clashes = []

        for l in tqdm(np.unique(self.data['length'].to_numpy()), desc="Calculating Ca-Ca metrics"):
            for s in np.unique(self.data['sample'].to_numpy()):
                structure = parser.get_structure('sample', os.path.join(self.path, f"length_{l}", f"sample_{s}", "sample.pdb"))

                #Get positions of Ca atoms
                ca_positions = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.has_id('CA'):
                                ca_positions.append(residue['CA'].get_coord())
                
                ca_ca_metrics = metrics.calc_ca_ca_metrics(np.array(ca_positions))
                ca_ca_deviation.append(ca_ca_metrics['ca_ca_deviation'])
                ca_ca_valid_percent.append(ca_ca_metrics['ca_ca_valid_percent'])
                num_ca_ca_clashes.append(ca_ca_metrics['num_ca_ca_clashes'])

        self.data['ca_ca_deviation'] = ca_ca_deviation
        self.data['ca_ca_valid_percent'] = ca_ca_valid_percent
        self.data['num_ca_ca_clashes'] = num_ca_ca_clashes

    def calc_scplddt(self):
        plddts = []
        for l in tqdm(np.unique(self.data['length'].to_numpy()), desc="Calculating SC pLDDT"):
            for s in np.unique(self.data['sample'].to_numpy()):

                data = pd.read_csv(os.path.join(self.path, f"length_{l}", f"sample_{s}", "self_consistency", "sc_results.csv"))
                idx = np.argmin(data['rmsd'].to_numpy())

                struct = bsio.load_structure(os.path.join(self.path, f"length_{l}", f"sample_{s}", "self_consistency", "esmf", f"sample_{idx}.pdb"), extra_fields=["b_factor"])
                plddt = round(struct.b_factor.mean(),2)
                plddts.append(plddt)
        
        self.data['scplddt'] = plddts

    def loc(self, length, sample=0):
        return self.data[(self.data['length'] == length) & (self.data['sample'] == sample)]

class DatasetMetrics(Metrics):

    def __init__(self, path, name="Dataset", force_update=False):
        super().__init__(path, name)

        if os.path.exists(os.path.join(path, 'metrics.csv')) and not force_update:
            self.data = pd.read_csv(os.path.join(path, 'metrics.csv'))
        else:
            data = {'pdb': [], 'length': []}

            for folder in tqdm(os.listdir(path), desc="Calculating lengths"):
                if os.path.isdir(f'{path}/{folder}'):
                    data['pdb'].append(folder)
                    data['length'].append(au.get_pdb_length(os.path.join(path, folder, 'sample.pdb')))
        
            self.data = pd.DataFrame(data)

    def calc_metrics(self):
        self.calc_scrmsd()
        self.calc_scplddt()
        self.calc_mdtraj_metrics()
        self.save()

    def calc_scrmsd(self):
        scrmsd = []
        non_coil_rmsd = []
        for pdb in tqdm(self.data['pdb'], desc="Calculating SC RMSD"):
            if not os.path.exists(os.path.join(self.path, pdb, "sc_results.csv")):
                scrmsd.append(None)
                continue

            data = pd.read_csv(os.path.join(self.path, pdb, "sc_results.csv"))
            scrmsd.append(min(data['rmsd'].tolist()))
            
            if 'non_coil_rmsd' in data.columns:
                non_coil_rmsd.append(min(data['non_coil_rmsd'].tolist()))
                
        self.data['scrmsd'] = scrmsd
        if len(non_coil_rmsd) > 0:
            if len(non_coil_rmsd) != len(scrmsd):
                raise ValueError(f"Non-coil RMSD not calculated for all samples: {len(non_coil_rmsd)} vs {len(scrmsd)}")
            self.data['non_coil_scrmsd'] = non_coil_rmsd

    def calc_scplddt(self):
        scplddt = []
        for pdb in tqdm(self.data['pdb'], desc="Calculating SC pLDDT"):
            if not os.path.exists(os.path.join(self.path, pdb, "sc_results.csv")):
                scplddt.append(None)
                continue

            data = pd.read_csv(os.path.join(self.path, pdb, "sc_results.csv"))
            idx = np.argmin(data['rmsd'].to_numpy())

            struct = bsio.load_structure(os.path.join(self.path, pdb, "esmf", f"sample_{idx}.pdb"), extra_fields=["b_factor"])
            plddt = round(struct.b_factor.mean(),2)
            scplddt.append(plddt)

        self.data['scplddt'] = scplddt

    def calc_mdtraj_metrics(self):
        helix_percent = []
        strand_percent = []
        coil_percent = []
        for pdb in tqdm(self.data['pdb'], desc="Calculating MDtraj metrics"):
            mdtraj_metrics = metrics.calc_mdtraj_metrics(os.path.join(self.path, pdb, "sample.pdb"))

            helix_percent.append(mdtraj_metrics['helix_percent'])
            strand_percent.append(mdtraj_metrics['strand_percent'])
            coil_percent.append(mdtraj_metrics['coil_percent'])

        self.data['helix_percent'] = helix_percent
        self.data['strand_percent'] = strand_percent
        self.data['coil_percent'] = coil_percent

    def calc_ca_ca_metrics(self):
        parser = Bio.PDB.PDBParser(QUIET=True)

        ca_ca_deviation = []
        ca_ca_valid_percent = []
        num_ca_ca_clashes = []

        for pdb in tqdm(self.data['pdb'], desc="Calculating CA-CA metrics"):
            structure = parser.get_structure('sample', os.path.join(self.path, pdb, "sample.pdb"))

            #Get positions of Ca atoms
            ca_positions = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.has_id('CA'):
                            ca_positions.append(residue['CA'].get_coord())
            
            ca_ca_metrics = metrics.calc_ca_ca_metrics(np.array(ca_positions))
            ca_ca_deviation.append(ca_ca_metrics['ca_ca_deviation'])
            ca_ca_valid_percent.append(ca_ca_metrics['ca_ca_valid_percent'])
            num_ca_ca_clashes.append(ca_ca_metrics['num_ca_ca_clashes'])

        self.data['ca_ca_deviation'] = ca_ca_deviation
        self.data['ca_ca_valid_percent'] = ca_ca_valid_percent
        self.data['num_ca_ca_clashes'] = num_ca_ca_clashes
    
    def add_plddt(self):
        plddts = []
        for pdb in tqdm(self.data['pdb'], desc="Getting pLDDT"):
            if not os.path.exists(os.path.join(self.path, pdb, 'plddt.txt')):
                plddts.append(None)
                continue
            with open(os.path.join(self.path, pdb, 'plddt.txt'), 'r') as f:
                plddt = np.array(json.load(f)[0]).astype(float)

            plddts.append(np.mean(plddt[:,2]))

        self.data['plddt'] = plddts

    def add_exptl(self):
        exptl = []
        for pdb in tqdm(self.data['pdb'], desc="Getting experimental method"):
            pdb = pdb[1:5]
            info = get_info(pdb)
            if info is not None:
                exptl.append(list(info['exptl'][0].values())[0])
            else:
                exptl.append(None)

        self.data['exptl'] = exptl