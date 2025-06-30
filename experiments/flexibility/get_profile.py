#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_smooth_array(length=128, end_width=10, peak_pos=[84], peak_width=[20]):
    # Initialize the array with base value
    arr = np.full(length, 0.5)
    
    # Apply the end values
    end_val_width = end_width
    if end_width > 0:
        arr[:end_val_width] = 2.5
        # arr[-end_val_width:] = 2.5
    
    # Insert peaks based on peak positions and widths
    for pos, width in zip(peak_pos, peak_width):
        arr[pos:pos+width] = 2.5
    
    # kernel_size = 5  # Defines the width of the moving average window
    # kernel = np.ones(kernel_size) / kernel_size  # Create a uniform averaging kernel
    # arr = np.convolve(arr, kernel, mode='same')
    return arr

#%%

# Generate the array with specified parameters
data = create_smooth_array(peak_pos=[35,80], peak_width=[15,15], end_width=0)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(data, label='Smoothed Custom Array')
plt.title('Smoothed Custom Array Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

print(len(data))
# %%
p = '../../flex_profiles/windows.txt'
Path(p).parent.mkdir(parents=True, exist_ok=True)

np.savetxt(p, data)
# %%
p = '../../flex_profiles/steps.txt'
Path(p).parent.mkdir(parents=True, exist_ok=True)

local_flex_base = np.full(128, 0.5)

local_flex = local_flex_base
local_flex[0:10] = 1.5
local_flex[10:30] = 0.7
local_flex[50:80] = 1.
local_flex[100:110] = 0.8

plt.plot(local_flex)

np.savetxt(p, local_flex)
# %%
