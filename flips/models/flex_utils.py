# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import torch
from torch.nn import functional as F
import importlib


from gafl.data import utils as du

def load_module(object):
    module, object = object.rsplit(".", 1)
    module = importlib.import_module(module)
    fn = getattr(module, object)
    return fn

def get_flex_embedding(values, max_bin_value=3, n_bins=6):
    """
    One-hot encodes local_flex of shape [batch_size, N] into a tensor of shape [batch_size, N, 8]
    where the last dimension one-hot encodes:
        - the scalar value for flexibility
        - the bin index (0-based)
        - the mask value (0, 1)
    Args:
        values: Tensor of scalar values of shape [N].
        max_bin_value: The maximum scalar value for the last bin.
        n_bins (int): Total number of bins.
    Returns:
        torch.Tensor: Tensor of shape [batch_size, N, 8]
    """
    # Ensure input is a float tensor, if not already
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, device="cuda:0")  # specify the device here
    values = values.float()
    # Just in case there are negative values != -1, set them to -1
    values = torch.where(values < 0, torch.tensor(-1, device=values.device), values)
    clamped_values = torch.clamp(values, min=0, max=max_bin_value)
    bin_edges = torch.linspace(start=0, end=max_bin_value, steps=n_bins, device=values.device)
    bins = torch.bucketize(clamped_values, bin_edges, right=True) - 1  # -1 to adjust to 0-based index
    one_hot = F.one_hot(bins, num_classes=n_bins)

    # where flex input is valid:
    mask_tensor = (values > 0).int()
    # scale clamped values to [0, 1] range
    # clamped_values = clamped_values / max_bin_value # NOTE: (this was not done in the pretrained checkpoint)
    result = torch.cat((clamped_values.unsqueeze(-1), one_hot, mask_tensor.unsqueeze(-1)), dim=-1)
    return result