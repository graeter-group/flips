"""
Generate a flexibility profile by drawing it with your mouse.
Has to be run with the jupyter interactive extension of vscode or copied in a notebook with the code between the #%% as individual cells.

Use label 'a' to draw the flexibility profile and label 'b' to draw a mask where values are not considered.
"""
#%%
from drawdata import ScatterWidget
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from pathlib import Path
import copy

widget = ScatterWidget()
widget

#%%

# modify the following parameters to adjust the flexibility profile
MAX_FLEX = 2
MIN_FLEX = 0.4
N_POINTS = 128
SIGMA = 1.5

def process(max_flex=2, min_flex=0.2, n_points=128, sigma=2):
    

    # Given data from your previous snippet
    data = widget.data
    x = [point['x'] for point in data if point['label'] == 'a']
    y = [point['y'] for point in data if point['label'] == 'a']

    # Convert lists to numpy arrays for easier manipulation
    x = np.array(x)
    y = np.array(y)

    # scale y:
    y = (y - y.min()) / (y.max() - y.min()) * (max_flex - min_flex) + min_flex

    # Sort the data by x-values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    max_x = copy.deepcopy(x_sorted[-1])
    x_sorted *= n_points / max_x # normalize x to (0, n_points)

    # Interpolate to create a new dataset with N_POINTS that are equally spaced
    x_new = np.arange(n_points)
    interpolator = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate")
    y_new = interpolator(x_new)


    # Apply Gaussian smoothing
    y_smoothed = gaussian_filter(y_new, sigma=sigma)

    # calculate the mask if given:
    mask_x = [point['x'] for point in data if point['label'] == 'b']
    if len(mask_x) > 0:
        mask_x = sorted(mask_x)
        mask_x = np.array(mask_x) * n_points / max_x
        # ceil/floor all values:
        mask_x_c = np.ceil(mask_x).astype(int).clip(0, n_points-1)
        mask_x_f = np.floor(mask_x).astype(int).clip(0, n_points-1)
        mask = np.zeros(n_points)
        mask[mask_x_c] = 1
        mask[mask_x_f] = 1

        mask = mask.astype(bool)
    else:
        mask = np.ones(n_points).astype(bool)

    # now set smoothed y to -1 where mask is 1:
    y_smoothed[mask == 0] = -1

    # Plot original and smoothed data
    plt.figure(figsize=(10, 5))
    plt.scatter(x_sorted, y_sorted, color='blue', label='Original Data')
    plt.plot(x_new[mask], y_smoothed[mask], color='red', label='Smoothed Data', linewidth=2)
    plt.xlim(0, n_points)
    plt.title('Drawn Flexibility Profile')
    plt.xlabel('Residue Index')
    plt.ylabel('Flexibility')
    plt.legend()
    plt.show()

    return y_smoothed

profile = process(max_flex=MAX_FLEX, min_flex=MIN_FLEX, n_points=N_POINTS, sigma=SIGMA)
# %%
profiledir = Path(__file__).parent.parent.parent / 'flex_profiles'
profiledir.mkdir(parents=True, exist_ok=True)
# np.savetxt(profiledir/'wetlab_.txt', profile)
# np.savetxt(profiledir/'drawn_profile_2.txt', profile)
# %%
