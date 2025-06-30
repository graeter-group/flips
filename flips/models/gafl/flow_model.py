# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Neural network architecture for the flow model."""
import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence, Tuple
import functools

from gafl.data import all_atom
from gafl.models import ipa_pytorch
from gafl.data import utils as du

from gatr.layers.linear import EquiLinear
from gatr.primitives.bilinear import geometric_product, _load_bilinear_basis
from gatr.primitives.dual import _compute_efficient_join
from gatr.primitives.invariants import norm
from gatr.utils.einsum import cached_einsum

from flips.models.edge_embedder import EdgeEmbedder
from flips.models.gafl.pga_utils import apply_versor, apply_inverse_versor, efficient_join, extract_motor, embed_frames, extract_frames, inf_norm, EquiLayerNorm
from flips.models.node_embedder import NodeEmbedder
from flips.models.gafl import pga_utils as pu

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def gfa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    gfa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        gfa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return  phi_angles


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s

class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed

class PairwiseGeometricBilinear(nn.Module):
    """Geometric bilinear layer between two different sets of multivectors.

    Pin-equivariant map between multivector tensors that constructs new geometric features via
    geometric products and the equivariant join (based on a reference vector).

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels of `x`
    out_mv_channels : int
        Output multivector channels
    hidden_mv_channels : int or None
        Hidden MV channels. If None, uses out_mv_channels.
    in_s_channels : int or None
        Input scalar channels of `x`. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    """

    def __init__(
        self,
        in_channels: int,
        
        out_channels: int,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Default options
        if hidden_channels is None:
            hidden_channels = out_channels

        out_channels_each = hidden_channels // 2
        assert (
            out_channels_each * 2 == hidden_channels
        ), "GeometricBilinear needs even channel number"

        self.linear_l = pu.MVLinear(in_channels, 2 * out_channels_each, splits=2)
        self.linear_r = pu.MVLinear(in_channels, 2 * out_channels_each, splits=2)
        self.out_channels_each = out_channels_each

        # Output linear projection
        self.linear_out = EquiLinear(
            hidden_channels, out_channels
        )

    def forward(
        self,
        mv1: torch.Tensor,
        mv2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        mv1 : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors argument 1
        mv2 : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors argument 2

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_mv_channels, 16)
            Output multivectors
        """
        
        l, _ = self.linear_l(mv1)
        r, _ = self.linear_r(mv2)

        gp_out = geometric_product(l[..., :self.out_channels_each, :], r[..., :self.out_channels_each, :])
        join_out = efficient_join(l[..., self.out_channels_each:, :], r[..., self.out_channels_each:, :])

        # Output linear
        out = torch.cat((gp_out, join_out), dim=-2)
        out, _ = self.linear_out(out)

        return out

class point_trafo(nn.Module):

    def __init__(self):
        super(point_trafo, self).__init__()

    def forward(self, x, T):
        return pu.apply_point_versor(x, T)

class ga_norm(nn.Module):

    def __init__(self):
        super(ga_norm, self).__init__()

    def forward(self, x):
        n = norm(x)[..., 0]
        inf_n = inf_norm(x)[..., 0]
        return n, inf_n

class ga_versor(nn.Module):

    def __init__(self):
        super(ga_versor, self).__init__()

    def forward(self, x, T):
        return apply_versor(x, T)
    
class ga_inverse_versor(nn.Module):

    def __init__(self):
        super(ga_inverse_versor, self).__init__()

    def forward(self, x, T):
        return apply_inverse_versor(x, T)
    
class global_rel_trafo(nn.Module):
    
    def __init__(self):
        super(global_rel_trafo, self).__init__()

    def forward(self, T):
        return pu.relative_global_frame_transformation(T)
    
class local_rel_trafo(nn.Module):
    
    def __init__(self):
        super(local_rel_trafo, self).__init__()

    def forward(self, T):
        return pu.relative_frame_transformation(T)
        
class GeometricFrameAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        gfa_conf,
        geometric_input=True,
        geometric_output=True,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(GeometricFrameAttention, self).__init__()
        self._gfa_conf = gfa_conf

        self.c_s = gfa_conf.c_s
        self.c_z = gfa_conf.c_z
        self.c_hidden = gfa_conf.c_hidden
        self.no_heads = gfa_conf.no_heads
        self.no_qk_points = gfa_conf.no_qk_points
        self.no_v_points = gfa_conf.no_v_points
        self.geometric_input = geometric_input
        self.geometric_output = geometric_output
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpk = self.no_heads * self.no_qk_points * 3
        self.linear_k_points = Linear(self.c_s, hpk)

        if self.geometric_input:
            self.linear_v_g = Linear(self.c_s, self.no_v_points * 16)
            self.merge_geometric = EquiLinear(2 * self.no_v_points, self.no_heads * self.no_v_points)
        else:
            self.linear_v_g = Linear(self.c_s, self.no_heads * self.no_v_points * 16)

        # self.merge_rel = EquiLinear(self.no_v_points + 1, self.no_v_points)

        self.g_layer_norm = EquiLayerNorm()

        self.bilinear_v = PairwiseGeometricBilinear(
            self.no_v_points, self.no_v_points
        )

        self.mbc_v = GMBCLayer(self.no_heads * self.no_v_points)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        # self.linear_T_rel = Linear(8, self.no_v_points * 8)

        self.head_weights = nn.Parameter(torch.zeros((gfa_conf.no_heads)))
        gfa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + 8 + self.no_v_points * 18 # 16 + 1 + 1
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        if self.geometric_output:
            self.geometric_out = EquiLinear(self.no_heads * self.no_v_points, self.no_v_points)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.point_trafo = point_trafo()
        self.ga_norm = ga_norm()
        self.ga_versor = ga_versor()
        self.ga_inverse_versor = ga_inverse_versor()
        self.local_rel_trafo = local_rel_trafo()

    def forward(
        self,
        s: torch.Tensor,
        g: torch.Tensor,
        z: Optional[torch.Tensor],
        T: torch.Tensor,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            T:
                [*, N_res, 16] versors representing the frames
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        T8 = pu.extract_motor(T)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)

        q_pts = self.point_trafo(q_pts, T8.unsqueeze(-2))

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 4)
        )

        # [*, N_res, H * P_q * 3]
        k_pts = self.linear_k_points(s)

        # [*, N_res, H * P_q, 3]
        k_pts = torch.split(k_pts, k_pts.shape[-1] // 3, dim=-1)
        k_pts = torch.stack(k_pts, dim=-1)

        k_pts = self.point_trafo(k_pts, T8.unsqueeze(-2))

        # [*, N_res, H, P_q, 3]
        k_pts = k_pts.view(k_pts.shape[:-2] + (self.no_heads, -1, 4))

        v_g = self.linear_v_g(s) 

        if self.geometric_input:
            v_g = v_g.view(*v_g.shape[:-1], self.no_v_points, 16)
            v_g, _ = self.merge_geometric(torch.cat([v_g, g], dim=-2))
            v_g = v_g.view(*v_g.shape[:-2], self.no_heads, self.no_v_points, 16)
        else:
            v_g = v_g.view(*v_g.shape[:-1], self.no_heads, self.no_v_points, 16)

        v_g = self.g_layer_norm(v_g)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4)[..., :3] - k_pts.unsqueeze(-5)[..., :3]
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1)
        
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        o_pt = self.ga_versor(v_g, T.unsqueeze(-2).unsqueeze(-3))
        o_pt = torch.sum(
        (
            a[..., None, :, :, None]
            * permute_final_dims(o_pt, (1, 3, 0, 2))[..., None, :, :]
        ),
        dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))

        o_pt = self.ga_inverse_versor(o_pt, T.unsqueeze(-2).unsqueeze(-3))

        v_rel = self.local_rel_trafo(T)
        v_rel = torch.matmul(a.transpose(-2, -3), v_rel)

        # o_pt, _ = self.merge_rel(torch.cat([o_pt, v_rel.unsqueeze(-2)], dim=-2))

        o_pt = self.bilinear_v(v_g, o_pt)

        # [*, N_res, H * P_v, 16]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 16) # TODO: Why reshape here?! -> Could move reshape somewhere else

        o_pt = self.mbc_v(o_pt)

        post_mbc_norm, post_mbc_infnorm = self.ga_norm(o_pt)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, N_res, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        v_rel = extract_motor(v_rel)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), *torch.unbind(v_rel, dim=-1), post_mbc_norm, post_mbc_infnorm, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )

        if self.geometric_output:
            g, _ = self.geometric_out(o_pt)
        else:
            g = None
        
        return s, g, v_rel
    
EPS = 1e-8 #TODO: Value may be changed

class BackboneUpdate(nn.Module):


    def __init__(self, c_s, c_g, c_T):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self.c_g = c_g
        self.c_T = c_T

        self.linear_hidden = Linear(self.c_s + 16 * self.c_g + 8 * self.c_T, 64)
        self.gelu = nn.GELU()

        # Choose initialization such that t=0 and R=1, i.e. the identiy map as update
        self.linear_final = Linear(64, 6, init="final")

    # def forward(self, s: torch.Tensor, T_rel: torch.Tensor, T: torch.Tensor, mask: torch.Tensor = None):
    def forward(self, s: torch.Tensor, g: torch.Tensor, T_rel:torch.Tensor, T: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            T:
                [*, N_res, c_T, 8] motors representing the frames
            g: 
                [*, N_res, c_g, 16] versors representing the frames
        Returns:
            [*, N_res, 6] update vector 
        """
        T_rel = T_rel.reshape(*T_rel.shape[:-2], -1)
        g = g.reshape(*g.shape[:-2], -1)

        in_ = torch.cat([s, g, T_rel], dim=-1)
        if mask is not None:
            in_ = in_ * mask

        update = self.linear_hidden(in_)
        update = self.gelu(update)
        update = self.linear_final(update)

        rot, trans = update[..., :3], update[..., 3:]

        R = torch.empty((*rot.shape[:-1], 4), device=rot.device, dtype=rot.dtype) 
        R[..., 1:] = rot
        R[..., 0] = 1.0
        rot = R / (torch.linalg.norm(R, dim=-1, keepdim=True) + EPS)

        T_update = pu.embed_rotor_t_frames(rot, trans)

        return geometric_product(T, T_update)


# Return list of slices, where each element in the list sclices the corresponding grade
def _grade_to_slice(dim):
    grade_to_slice = list()
    subspaces = torch.as_tensor([math.comb(dim, i) for i in range(dim + 1)])
    for grade in range(dim + 1):
        index_start = subspaces[:grade].sum()
        index_end = index_start + math.comb(4, grade)
        grade_to_slice.append(slice(index_start, index_end))
    return grade_to_slice

@functools.lru_cache()
def bilinear_product_paths(type='gmt'):
    dim = 4
    grade_to_slice = _grade_to_slice(dim)
    gp_paths = torch.zeros((dim + 1, dim + 1, dim + 1), dtype=bool)

    if type == 'gmt':
        mt = _load_bilinear_basis('gp')
    elif type == 'jmt':
        mt = _compute_efficient_join()

    for i in range(dim + 1):
        for j in range(dim + 1):
            for k in range(dim + 1):
                s_i = grade_to_slice[i]
                s_j = grade_to_slice[j]
                s_k = grade_to_slice[k]

                m = mt[s_i, s_j, s_k]
                gp_paths[i, j, k] = (m != 0).any()

    return gp_paths

# Geometric many body contraction layer currently only implemeted for n=3
class GMBCLayer(nn.Module):

    def __init__(self, num_features):
        """
        Args:
            num_features:
                Number of output features
        """

        super(GMBCLayer, self).__init__()
        # TODO: Refactor everything such that register buffer is not needed here
        self.register_buffer("subspaces", torch.as_tensor([math.comb(4, i) for i in range(4 + 1)])) # Register as buffer so that it is moved to gpu with the model
        self.num_features = num_features

        self.register_buffer("gp", bilinear_product_paths('gmt'))
        self.register_buffer("jp", bilinear_product_paths('jmt'))

        self.gp_weights = nn.Parameter(torch.empty(num_features, self.gp.sum()))
        self.jp_weights = nn.Parameter(torch.empty(num_features, self.jp.sum()))

        self.linear = pu.MVLinear(num_features, 2*num_features, splits=2)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.gp_weights, std=1 / (math.sqrt(4 + 1)))
        torch.nn.init.normal_(self.jp_weights, std=1 / (math.sqrt(4 + 1)))

    def _get_weight(self):
        gp_weights = torch.zeros(
            self.num_features,
            *self.gp.size(),
            dtype=self.gp_weights.dtype,
            device=self.gp_weights.device,
        )
        gp_weights[:, self.gp] = self.gp_weights
        gp_weights_repeated = (
            gp_weights.repeat_interleave(self.subspaces, dim=-3)
            .repeat_interleave(self.subspaces, dim=-2)
            .repeat_interleave(self.subspaces, dim=-1)
        )

        jp_weights = torch.zeros(
            self.num_features,
            *self.jp.size(),
            dtype=self.jp_weights.dtype,
            device=self.jp_weights.device,
        )
        jp_weights[:, self.jp] = self.jp_weights
        jp_weights_repeated = (
            jp_weights.repeat_interleave(self.subspaces, dim=-3)
            .repeat_interleave(self.subspaces, dim=-2)
            .repeat_interleave(self.subspaces, dim=-1)
        )
        return _load_bilinear_basis('gp', dtype=self.gp_weights.dtype, device=self.gp_weights.device) * gp_weights_repeated + _compute_efficient_join(dtype=self.gp_weights.dtype, device=self.gp_weights.device) * jp_weights_repeated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_v] single representation
        Returns:
            [*, C_s] single representation update
        """
        W = self._get_weight()
        x, _ = self.linear(x)
        return cached_einsum('nijk,...nj,...nk->...ni', W, x[..., :self.num_features, :], x[..., self.num_features:, :]) + x[..., self.num_features:, :]

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()

        self._model_conf = model_conf
        self._gfa_conf = model_conf.gfa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._gfa_conf.num_blocks):
            self.trunk[f'gfa_{b}'] = GeometricFrameAttention(self._gfa_conf, geometric_input=bool(b), geometric_output=True)
            self.trunk[f'gfa_ln_{b}'] = nn.LayerNorm(self._gfa_conf.c_s)
            tfmr_in = self._gfa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._gfa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._gfa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, self._gfa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._gfa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(
                self._gfa_conf.c_s,
                self._gfa_conf.no_v_points,
                self._gfa_conf.no_heads,)

            if b < self._gfa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._gfa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_feats['t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        local_flex = input_feats['local_flex']
        
        # modify node_embedder to delete mask_encode; now it's in there
        init_node_embed = self.node_embedder(continuous_t, node_mask, local_flex)
        
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask)

        # Initial rigids
        # curr_rigids = du.create_rigid(rotmats_t, trans_t,)
        curr_frames = embed_frames(rotmats_t, trans_t * du.ANG_TO_NM_SCALE)

        # Main trunk
        # curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        g = None
        for b in range(self._gfa_conf.num_blocks):
            gfa_embed, g, gfa_rel = self.trunk[f'gfa_{b}'](
                node_embed,
                g,
                edge_embed,
                curr_frames,
                node_mask)
            gfa_embed *= node_mask[..., None]
            gfa_rel = gfa_rel * node_mask[..., None, None]
            if g is not None:
                g =  g * node_mask[..., None, None]
            node_embed = self.trunk[f'gfa_ln_{b}'](node_embed + gfa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)    
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            curr_frames = self.trunk[f'bb_update_{b}'](
                node_embed,
                g,
                gfa_rel,
                curr_frames,
                node_mask[..., None],)

            if b < self._gfa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        pred_rotmats, pred_trans = extract_frames(curr_frames)
        pred_trans = pred_trans * du.NM_TO_ANG_SCALE
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
        }