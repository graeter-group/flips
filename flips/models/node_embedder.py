# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Neural network for embedding node features."""
import torch
from torch import nn
from gafl.models.utils import get_index_embedding, get_time_embedding, get_length_embedding

from flips.models.flex_utils import get_flex_embedding


class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)

        if "total_length_emb_dim" in self._cfg:
            self.total_length_emb_dim = self._cfg.total_length_emb_dim
        else:
            self.total_length_emb_dim = 0
        
        if "total_flex_emb_dim" in self._cfg:
            self.total_flex_emb_dim = self._cfg.total_flex_emb_dim
        else:
            self.total_flex_emb_dim = 0

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask, local_flex=None):
        """Encodes node features, including diffuse_mask."""
        
        # shape [b, n_res, dev]
        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        if local_flex is None:
            local_flex = torch.zeros((b, num_res), dtype=torch.float32, device=device)

        pos = torch.arange(num_res, dtype=torch.float32, device=device).unsqueeze(0)

        # Get position embedding
        pos_emb = get_index_embedding(
            pos, self.c_pos_emb - self.total_length_emb_dim - self.total_flex_emb_dim, max_len=2056
        ).repeat(b, 1, 1)

        # Append length embedding (if used)
        if self.total_length_emb_dim > 0:
            length_embedding = get_length_embedding(pos, embed_size=self.total_length_emb_dim, max_len=2056)
            length_embedding = length_embedding.repeat(b, 1, 1)
            pos_emb = torch.cat([pos_emb, length_embedding], dim=-1)

        # Append flexibility embedding (if used)
        if self.total_flex_emb_dim > 0:
            flex_emb = get_flex_embedding(values=local_flex, max_bin_value=3, n_bins=self.total_flex_emb_dim - 2)
            pos_emb = torch.cat([pos_emb, flex_emb], dim=-1)

        # Apply mask to position embedding
        pos_emb = pos_emb * mask.unsqueeze(-1)
        input_feats = [pos_emb, self.embed_t(timesteps, mask)]

        # Final concatenation and projection
        return self.linear(torch.cat(input_feats, dim=-1))