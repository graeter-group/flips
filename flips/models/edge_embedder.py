# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn

from gafl.models.utils import get_index_embedding, calc_distogram

class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        if "permutation_equivariance" in self._cfg:
            self.permutation_equivariance = self._cfg.permutation_equivariance
        else:
            self.permutation_equivariance = False

        if "embed_breaks" in self._cfg:
            self.embed_breaks = self._cfg.embed_breaks
        else:
            self.embed_breaks = False

        if self.embed_breaks:
            self.c_breaks = 2
        else:
            self.c_breaks = 0

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos, breaks=None):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = get_index_embedding(rel_pos, self._cfg.feat_dim - self.c_breaks, max_len=2056)

        # Adding breaks here enables to keep the same feat_dim and thus the same parameter set as without breaks thus making both models backwards compatible
        if self.embed_breaks:
            if breaks is None:
                breaks = torch.zeros([pos.shape[0], pos.shape[1], pos.shape[1]], device=pos.device, dtype=torch.float32)
            else:
                breaks = breaks[:, :, None] * breaks[:, None, :] * (torch.diag(torch.ones(pos.shape[1] - 1, device=pos.device, dtype=torch.float32), diagonal=1) + torch.diag(torch.ones(pos.shape[1] - 1, device=pos.device, dtype=torch.float32), diagonal=-1)).unsqueeze(0)

            pos_emb = torch.cat([pos_emb, breaks.unsqueeze(-1), (1 - breaks).unsqueeze(-1)], dim=-1)

        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, res_idx=None, breaks=None, res_mask=None):
        num_batch, num_res, _ = s.shape
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        if res_idx is None:
            pos = torch.arange(
                num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        else:
            pos = res_idx.float()
        
        if self.permutation_equivariance:
            # NOTE: remove positional encoding, except for first and last residue
            pos[:, 1:-1] = torch.ones_like(pos[:, 1:-1])
            # set the first residue to 0 and the last to 2, then multiply by half of max len
            pos[:, 0] = 0
            pos[:, -1] = 2
            pos = pos * 1028

        relpos_feats = self.embed_relpos(pos, breaks=breaks)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = torch.concat([cross_node_feats, relpos_feats, dist_feats, sc_feats], dim=-1)
        # embedding diffuse_mask
        # if self._cfg.embed_diffuse_mask:
        #     diff_feat = self._cross_concat(res_mask[..., None], num_batch, num_res)
        #     all_edge_feats = torch.cat([all_edge_feats, diff_feat], dim=-1)
        
        edge_feats = self.edge_embedder(all_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats