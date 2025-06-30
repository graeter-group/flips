#%%
# make the dataloader:
from gafl_flex.data.pdb_dataloader import PdbDataset
from data import protein
from data import utils

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
# %%

dataset_cfg = OmegaConf.load("../../configs/data/flexibility.yaml").dataset
dataset_cfg.min_num_res = 128
dataset_cfg.max_num_res = 128

outpath = Path(__file__).parent / "pdb_files_scope"
outpath.mkdir(exist_ok=True)
local_flex_profiles = []
pdb_names = []
# %%
ds = PdbDataset(dataset_cfg=dataset_cfg, is_training=True)
for entry in ds:
    local_flex = entry['local_flex']
    pdb_name = entry['pdb_name']
    local_flex_profiles.append(local_flex.cpu().numpy())
    pdb_names.append(pdb_name)
#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_8_profiles(profiles, names):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.set_title(f'{i}: {names[i]}')
        ax.plot(profiles[i])
    plt.show()
# %%
# make batches of 8:
n = 8
n_batches = len(local_flex_profiles) // n
for i in range(n_batches):
    profiles = local_flex_profiles[i*n:(i+1)*n]
    names = pdb_names[i*n:(i+1)*n]
    plot_8_profiles(profiles, names)
# %%
pdbnames = ['d2i4ka_', 'd6m6ea_', 'd2bkma_', 'd4qspa1']
mask = np.arange(128) < 60

chosen_profiles = []
for pdbname in pdbnames:
    idx = pdb_names.index(pdbname)
    p = local_flex_profiles[idx]
    p[mask] = -1
    chosen_profiles.append(p)

for i, profile in enumerate(chosen_profiles):
    plt.plot(profile, label=pdbnames[i])
    plt.title(f'scope_{i}')
    plt.show()
    Path(f'../../flex_profiles/scope_{i}.txt').parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(f'../../flex_profiles/scope_{i}.txt', profile)
# %%
