# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Licensed under the MIT license.

import torch
import numpy as np

from functools import lru_cache

from gatr.primitives.linear import reverse, _compute_reversal, NUM_PIN_LINEAR_BASIS_ELEMENTS, equi_linear
from gatr.primitives.bilinear import _load_bilinear_basis, geometric_product
from gatr.primitives.dual import _compute_efficient_join
from gatr.utils.einsum import cached_einsum
from gatr.interface.rotation import embed_rotation, extract_rotation
from gatr.interface.translation import embed_translation, extract_translation
from gatr.primitives.invariants import compute_inner_product_mask, norm
from gatr.interface.scalar import embed_scalar

from gafl.data.so3_utils import rotmat_to_rotquat, rotquat_to_rotmat

MOTOR_DIMS = [0,5,6,7,8,9,10,15]

def embed_rotor(rotor: torch.Tensor) -> torch.Tensor:
    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = rotor.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=rotor.dtype, device=rotor.device)

    multivector[..., 0] = rotor[..., 0]
    multivector[..., 8] = rotor[..., 1]
    multivector[..., 9] = rotor[..., 2]  
    multivector[..., 10] = rotor[..., 3]

    return multivector

def extract_rotor(multivector: torch.Tensor, normalize=False) -> torch.Tensor:
    rotor = torch.cat(
        [
            multivector[..., [0]],
            multivector[..., [8]],
            multivector[..., [9]],
            multivector[..., [10]],
        ],
        dim=-1,
    )

    if normalize:
        rotor = rotor / (torch.linalg.norm(rotor, dim=-1, keepdim=True) + 1e-8)

    return rotor

def embed_rotor_t_frames(R, t):
    """
    Computes the frame embedding in PGA of a euclidean transformation T = (R, t).
    Args:
        R: Rotor, shape: [*, N_res, 4]
        t: Translation vector, shape: [*, N_res, 3]
    Returns:
        v: Frame embedding, shape: [*, N_res, 16]
    """
    
    rotmv = embed_rotor(R)
    transmv = embed_translation(t)

    return geometric_product(transmv, rotmv)

def extract_rotor_t_frames(v):
    rot_signs = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, -1 , -1, -1, 0, 0, 0, 0, 0], dtype=v.dtype, device=v.device)
    rotmv_inv = v * rot_signs

    t = extract_translation(geometric_product(v, rotmv_inv)) #Again the "-" compensates for the gatr convention
    R = extract_rotor(v)

    return R, t

def embed_frames(R, t):
    """
    Computes the frame embedding in PGA of a euclidean transformation T = (R, t).
    Args:
        R: Rotation matrix, shape: [*, N_res, 3, 3]
        t: Translation vector, shape: [*, N_res, 3]
    Returns:
        v: Frame embedding, shape: [*, N_res, 16]
    """
    
    rotquat = rotmat_to_rotquat(R)
    # frameflow lib uses a different quaternion convention than gatr
    rotmv = embed_rotation(rotquat[...,[1,2,3,0]]) #TODO: Maybe merge into single function
    transmv = embed_translation(t) #gatr has a strange "-" sign in its implementation

    return geometric_product(transmv, rotmv)

def embed_quat_frames(Q, t):
    """
    Computes the frame embedding in PGA of a euclidean transformation T = (Q, t).
    Args:
        Q: Quaternion, shape: [*, N_res, 4]
        t: Translation vector, shape: [*, N_res, 3]
    Returns:
        v: Frame embedding, shape: [*, N_res, 16]
    """
    
    rotmv = embed_rotation(Q)
    transmv = embed_translation(t)

    return geometric_product(transmv, rotmv)

def extract_frames(v):
    """
    Extracts the rotation matrix and translation vector from the frame embedding in PGA.
    Args:
        v: Frame embedding, shape: [*, N_res, 16]
    Returns:
        R: Rotation matrix, shape: [*, N_res, 3, 3]
        t: Translation vector, shape: [*, N_res, 3]
    """

    rot_signs = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, -1 , -1, -1, 0, 0, 0, 0, 0], dtype=v.dtype, device=v.device)
    rotmv_inv = v * rot_signs

    t = extract_translation(geometric_product(v, rotmv_inv)) #Again the "-" compensates for the gatr convention

    rotquat = extract_rotation(v)
    #Convert to gatr convention
    rotquat = rotquat[...,[3,0,1,2]]
    R = rotquat_to_rotmat(rotquat)
    return R, t

def reverse_versor(v):
    return _compute_reversal(device=v.device, dtype=v.dtype)[MOTOR_DIMS] * v

def apply_versor(x, v):

    """
    Applies a versor to a multivector.
    Args:
        x: Multivector to be transformed, shape: [*, 16]
        v: Versor, shape: [*, 16]
    Returns:
        y: Transformed multivector, shape: [*, 16]
    """

    v_inv = reverse(v)
    y = geometric_product(v, geometric_product(x, v_inv))

    return y


def apply_inverse_versor(x, v):
    
    """
    Applies the inverse of a versor to a multivector.
    Args:
        x: Multivector to be transformed, shape: [*, 16]
        v: Versor, shape: [*, 16], Make sure that this is properly normalized!
    Returns:
        y: Transformed multivector, shape: [*, 16]
    """

    v_inv = reverse(v)
    y = geometric_product(v_inv, geometric_product(x, v))

    return y

# @lru_cache()
# def load_point_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
#     return _load_bilinear_basis("gp", device=device, dtype=dtype)[:, 11:15, MOTOR_DIMS]

# @lru_cache()
# def load_reversed_point_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
#     return _load_bilinear_basis("gp", device=device, dtype=dtype)[11:15, MOTOR_DIMS, :]

@lru_cache()
def load_point_versor_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:

    gp = _load_bilinear_basis("gp", device=device, dtype=dtype)
    # c_ijk c_klm v_j x_l v_inv_m -> c'_ijlm v_j x_l v_inv_m
    # v_inv = reversal * v (r_ij * v_j)
    # c''_ijln = c'ijlm r_mn
    c = cached_einsum("ijk, klm -> ijlm", gp[11:15, MOTOR_DIMS, :], gp[:, 11:15, MOTOR_DIMS])
    r = _compute_reversal(device=device, dtype=dtype)[MOTOR_DIMS]
    return c * r

def apply_point_versor(x, v, threshold=1e-3):

    mv = torch.empty(x.shape[:-1] + (4,), device=x.device, dtype=x.dtype)
    mv[..., 3] = 1.0
    mv[..., 2] = -x[..., 0]  # x-coordinate embedded in x_023
    mv[..., 1] = x[..., 1]  # y-coordinate embedded in x_013
    mv[..., 0] = -x[..., 2]  # z-coordinate embedded in x_012

    # # Compute geometric product
    # mv = cached_einsum("i j k, ... j, ... k -> ... i", load_point_kernel(device=x.device, dtype=x.dtype), mv, reverse_versor(v))
    # mv = cached_einsum("i j k, ... j, ... k -> ... i", load_reversed_point_kernel(device=x.device, dtype=x.dtype), v, mv)

    mv = cached_einsum("i j k l, ... j, ... k, ... l -> ... i", load_point_versor_kernel(device=x.device, dtype=x.dtype), v, mv, v)
    return mv
    
    # return 'torch.cat([-mv[..., [2]], mv[..., [1]], -mv[..., [0]]], dim=-1)'

############################################################################################################
# The alternative versions of versor application below are in fact slower for the full geometric product
# The idea can still be used when applying versors to point (see above) as the required kernel is much smaller
# In the above case we can contract over the largest dimension which leaves a kernel of size 4 x 8 x 4 x 8 = 1024
# compared to the gp kernel of size 16 x 16 x 16 = 4096 which is needed twice for versor application
# For the full geometric product the kernel would be of size 16 x 16 x 16 x 16 = 65536
############################################################################################################

# @lru_cache()
# def load_versor_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:

#     gp = _load_bilinear_basis("gp", device=device, dtype=dtype)
#     # c_ijk c_klm v_j x_l v_inv_m -> c'_ijlm v_j x_l v_inv_m
#     # v_inv = reversal * v (r_ij * v_j)
#     # c''_ijln = c'ijlm r_mn
#     c = cached_einsum("ijk, klm -> ijlm", gp, gp)
#     r = _compute_reversal(device=device, dtype=dtype)
#     return c * r

# @lru_cache()
# def load_inverse_versor_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:

#     gp = _load_bilinear_basis("gp", device=device, dtype=dtype)
#     # c_ijk c_klm v_inv_j x_l v_m -> c'_ijlm v_inv_j x_l v_m
#     # v_inv = reversal * v (r_ij * v_j)
#     # c''_inlm = c'ijlm r_jn	
#     c = cached_einsum("ijk, klm -> ijlm", gp, gp)
#     r = _compute_reversal(device=device, dtype=dtype)
#     return c * r[None, :, None, None]

# def fast_versor(x, v) -> torch.Tensor:
#     # Select kernel on correct device
#     c = load_versor_kernel(device=x.device, dtype=x.dtype)
#     return cached_einsum("ijlm, ... j, ... l, ... m -> ... i", c, v, x, v)

# def fast_inverse_versor(x, v) -> torch.Tensor:
#     # Select kernel on correct device
#     c = load_inverse_versor_kernel(device=x.device, dtype=x.dtype)
#     return cached_einsum("ijlm, ... j, ... l, ... m -> ... i", c, v, x, v)


#TODO: Compare against using the normal geometric product with broadcastable input vectors
def cross_gp(x: torch.Tensor, y: torch.Tensor):
    """
    Parameters
    ----------
    x : torch.Tensor with shape (..., N_res, 16)

    y : torch.Tensor with shape (..., N_res, 16)

    Returns
    -------
    outputs : torch.Tensor with shape (..., N_res, N_res, 16)
    """

    # Select kernel on correct device
    gp = _load_bilinear_basis("gp", device=x.device, dtype=x.dtype)

    # Compute geometric product
    outputs = cached_einsum("i j k, ... nj, ... mk -> ... nmi", gp, x, y)

    return outputs

def relative_global_frame_transformation(v):
      
    v_inv = reverse(v)
    v_global = cross_gp(v, v_inv)
    
    return v_global

def relative_frame_transformation(v):
    """
    Computes invariant and equivariant representations of relative frame transformations between all pairs of frames.
    Args:
        v: Frame representations for all residues, shape: [*, N_res, 16]
        Assume normalized frames v ~v = 1
        #TODO: Change tensor shape to [*, N_res, 8] (drop all zero components)

    Returns:
        v_local: Invariant relative frame transformations, shape: [*, N_res, N_res, 16]
    """
        
    v_inv = reverse(v)
    v_local = cross_gp(v_inv, v)
    
    return v_local

# When working with local frames, we do not need to ensure equivariance of the join using a reference multivector
def efficient_join(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Computes the join, using the efficient implementation.

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.

    Returns
    -------
    outputs : torch.Tensor
        Equivariant join result.
    """

    kernel = _compute_efficient_join(x.device, x.dtype)
    return cached_einsum("i j k , ... j, ... k -> ... i", kernel, x, y)

def extract_motor(v):
    """
    Extracts the motor from a frame embedding in PGA.
    Args:

        v: Frame embedding, shape: [*, 16]
    Returns:
        m: Motor, shape: [*, 8]
    """
    return v[..., MOTOR_DIMS]

def embed_motor(m):
    """
    Embeds a motor in a frame embedding in PGA.
    Args:
        m: Motor, shape: [*, 8]
    Returns:
        v: Frame embedding, shape: [*, 16]
    """

    v = torch.zeros(m.shape[:-1] + (16,), device=m.device, dtype=m.dtype)
    v[..., MOTOR_DIMS] = m
    return v

@lru_cache()
def compute_inf_norm_mask(device=torch.device("cpu")) -> torch.Tensor:
    # Invert boolean inner product mask
    return ~compute_inner_product_mask(device=device)

def inf_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x[..., compute_inf_norm_mask(device=x.device)], dim=-1, keepdim=True)


from typing import Tuple
from torch import nn

def equi_layer_norm(
    x: torch.Tensor, channel_dim: int = -2, epsilon: float = 0.01
) -> torch.Tensor:

    # Compute mean_channels |inputs|^2
    squared_norms = x[..., compute_inner_product_mask(device=x.device)].pow(2).sum(dim=-1, keepdim=True)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    squared_norms = torch.clamp(squared_norms, epsilon)

    # Rescale inputs
    outputs = x / torch.sqrt(squared_norms)

    return outputs

class EquiLayerNorm(nn.Module):

    def __init__(self, mv_channel_dim=-2, epsilon: float = 0.01):
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon

    def forward(
        self, multivectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )

        return outputs_mv
    
class MVLinear(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, splits=1):
        super().__init__()
        assert out_channels % splits == 0

        weights = []
        for i in range(splits):
            weights.append(
                torch.empty((out_channels // splits, in_channels, NUM_PIN_LINEAR_BASIS_ELEMENTS))
            )

        mv_component_factors, mv_factor = self._compute_init_factors(1.0, 1.0 / np.sqrt(3.0), True)

        # Let us fist consider the multivector outputs.
        fan_in = in_channels
        bound = mv_factor / np.sqrt(fan_in)
        for i in range(splits):
            for j, factor in enumerate(mv_component_factors):
                nn.init.uniform_(weights[i][..., j], a=-factor * bound, b=factor * bound)

        concatenated_weights = torch.cat(weights, dim=0)
        self.weight = nn.Parameter(concatenated_weights)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_channels, 1)))
        else:
            self.bias = None

    def forward(
        self, multivectors: torch.Tensor
    ):
        outputs_mv = equi_linear(multivectors, self.weight)  # (..., out_channels, 16)

        if self.bias is not None:
            bias = embed_scalar(self.bias)
            outputs_mv = outputs_mv + bias

        return outputs_mv, None

    @staticmethod
    def _compute_init_factors(gain, additional_factor, use_mv_heuristics):
        """Computes prefactors for the initialization.

        See self.reset_parameters().
        """
        mv_factor = gain * additional_factor * np.sqrt(3)

        # Individual factors for each multivector component
        if use_mv_heuristics:
            mv_component_factors = torch.sqrt(
                torch.Tensor([1.0, 4.0, 6.0, 2.0, 0.5, 0.5, 1.5, 1.5, 0.5])
            )
        else:
            mv_component_factors = torch.ones(NUM_PIN_LINEAR_BASIS_ELEMENTS)
        return mv_component_factors, mv_factor