# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from typing import Tuple
from scipy.spatial.transform import Rotation
import copy
from scipy.optimize import linear_sum_assignment
import logging

from gafl.data import so3_utils
from gafl.data import all_atom
from gafl.data import utils as du
from backflip.deployment.inference_class import BackFlip


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    # 1 - diffuse_mask: 0 for motifs then and none for scaffold
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self._log = logging.getLogger(__name__)

        BACKWARDS_COMPATIBLITITY = True

        if BACKWARDS_COMPATIBLITITY: # apply old config settings if new keys are not present
            if hasattr(self._cfg, 'noise_res_scaling_power'):
                self.noise_res_scaling_power = self._cfg.noise_res_scaling_power
            else:
                self.noise_res_scaling_power = 0

            if hasattr(self._cfg, 'noise_scale'):
                self.noise_scale = self._cfg.noise_scale
            else:
                self.noise_scale = 1.0

            if hasattr(self._cfg, 'after_ot_noise'):
                self.after_ot_noise = self._cfg.after_ot_noise
            else:
                self.after_ot_noise = False

            if hasattr(self._cfg, 'instance_ot'):
                self.instance_ot = self._cfg.instance_ot
            else:
                self.instance_ot = False

            if hasattr(self._cfg, 'igso3_training'):
                self.igso3_training = self._cfg.igso3_training
            elif self.instance_ot:
                self.igso3_training = False
            else:
                self.igso3_training = True

            if hasattr(self._cfg, 't_interval'):
                self.t_interval = self._cfg.t_interval
            else:
                self.t_interval = [0, 1]

            if hasattr(self._cfg, 't_sampling_focus'):
                self.t_sampling_focus = self._cfg.t_sampling_focus
            else:
                self.t_sampling_focus = 0.

            # Classifier free guidance config
            if hasattr(self._cfg, "classifier_free_guidance"):
                cfg_config = self._cfg.classifier_free_guidance
                self.classifier_free_guidance_scale = getattr(cfg_config, "cfg_scale", 1.0)
                self._log.info("Classifier free guidance scale set to %f", self.classifier_free_guidance_scale)
                if self.classifier_free_guidance_scale not in [0., 1.]:
                    raise NotImplementedError("Classifier free guidance scale must be 0 or 1 currently.")
            else:
                self._log.info("No classifier free guidance config passed, defaulting to 1.0 (conditional sampling).")
                self.classifier_free_guidance_scale = 1.0

            # Classifer guidance config
            if not hasattr(self._cfg, "classifier_guidance"):
                self._log.info("No classifier guidance config passed, inference without classifier guidance.")
                self.classifier_guidance = False
                self.classifier_guidance_scale = 0.0
            else:
                cg_cfg = self._cfg.classifier_guidance
                self.classifier_guidance_scale = getattr(cg_cfg, "cg_scale", 5.0)
                if self.classifier_guidance_scale == 0:
                    self._log.info("Classifier guidance scale is 0, inference without classifier guidance.")
                    self.classifier_guidance = False
                else:
                    self.classifier_guidance = True

                self.cg_schedule = getattr(cg_cfg, "cg_schedule", "static")
                self.length_scaling = getattr(cg_cfg, "length_scaling", False)

                if self.length_scaling and self.classifier_guidance:
                    self._log.info("Length scaling enabled, overriding classifier guidance scale to 5.0")
                    self.classifier_guidance_scale = 5.0

        else:
            self.noise_res_scaling_power = self._cfg.noise_res_scaling_power
            self.noise_scale = self._cfg.noise_scale
            self.after_ot_noise = self._cfg.after_ot_noise
            self.instance_ot = self._cfg.instance_ot

        if self.instance_ot:
            raise NotImplementedError("Instance OT is not implemented")
        
        if self.classifier_guidance_scale != 0:
            self._log.info("Starting inference with classifier guidance. cg_scale set to %f", self.classifier_guidance_scale)
            self.backflip = BackFlip.from_tag('backflip-0.2', device='cuda').model


    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device
        if self.instance_ot:
            self.permuter.to(self._device)

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)

        # transform the uniform distribution to the mode distribution
        s = self.t_sampling_focus
        if s != 0:
            t = self.f_mode_transformation(t, s)

        # Map to training interval
        t = self.t_interval[0] + t * (self.t_interval[1] - self.t_interval[0])

        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t
    
    @staticmethod
    def f_mode_transformation(t, s):
        """
        Transformation to the mode distribution from https://arxiv.org/abs/2403.03206
        """
        # Ensure s is a float, and t is a torch.Tensor
        s = float(s)  
        u = 1 - t - s * (torch.cos(torch.pi / 2 * t) ** 2 - 1 + t)
        return u

    def _corrupt_trans(self, noisy_batch):
        trans_nm_0 = _centered_gaussian(*noisy_batch['res_mask'].shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 *= self.noise_scale

        # make the noise blob larger the longer the protein is:
        if self.noise_res_scaling_power > 0:
            trans_0 *= (trans_0.shape[1] / 128.)**self.noise_res_scaling_power

        trans_0 = self._batch_ot(trans_0, noisy_batch['trans_1'], noisy_batch['res_mask'])
        
        if self.instance_ot:
            noisy_batch['trans_0'] = trans_0
            # reorders trans_0 according to the optimal permutation
            noisy_batch = _ot_permutation(noisy_batch=noisy_batch)

            # kabsch align again (lazy: could be done better)
            # (this also allows to interachange batches, which will probably not happen due to the node optimal transport. its just in here for simplicity, we actually only need the kabsch alignment)
            trans_0 = self._batch_ot(noisy_batch['trans_0'], noisy_batch['trans_1'], noisy_batch['res_mask'])

        # add more noise to teach the model that it should move the residues more:
        if self.after_ot_noise > 0:
            trans_nm_0_add = _centered_gaussian(*noisy_batch['res_mask'].shape, self._device)
            trans_0_add = trans_nm_0_add * du.NM_TO_ANG_SCALE
            trans_0_add *= self.after_ot_noise
            trans_0 += trans_0_add
        
        noisy_batch['trans_0'] = trans_0

        t = noisy_batch['t']
        trans_1 = noisy_batch['trans_1']
        res_mask = noisy_batch['res_mask']
        
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        trans_t = trans_t * res_mask[..., None]

        noisy_batch['trans_t'] = trans_t

        return noisy_batch
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        ############################
        # NOTE: use uniform noise instead of igso3
        # noisy_rotmats = self.igso3.sample(
        #     torch.tensor([1.5]),
        #     num_batch*num_res
        # ).to(self._device)
        if self.igso3_training:
            noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch*num_res).to(self._device)
            noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        else:
            noisy_rotmats = _uniform_so3(num_batch, num_res, self._device)
        ############################

        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch, t=None):
        noisy_batch = copy.deepcopy(batch)
        num_batch, _ = batch['res_mask'].shape

        # [B, 1]
        if t is None:
            t = self.sample_t(num_batch)[:, None]
        else:
            t = torch.ones(num_batch, 1, device=self._device, dtype=torch.float32) * t
        noisy_batch['t'] = t

        # Apply corruptions
        noisy_batch = self._corrupt_trans(noisy_batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)
    
    def cg_scheduler(self, t, t_max, scale_max, schedule:str='linear', alpha:int=1):
        if schedule == 'linear':
            return scale_max * (t / t_max)**alpha
        elif schedule == 'linear_decay':
            return scale_max * (1 - t / t_max)**alpha
        elif schedule == 'static':
            return scale_max
        else:
            logging.info(f"Unknown schedule {schedule}, using static schedule")
            return scale_max

    def sample(
            self,
            num_batch,
            num_res,
            model,
            prior_sample:Tuple[torch.Tensor, torch.Tensor]=None,
            save_path=None,
            local_flex=None,
            flex_model=None,
            flex_result_ptr=None,
        ):

        res_mask = torch.ones(num_batch, num_res, device=self._device)
        batch = {
            'res_mask': res_mask,
        }

        if prior_sample is None:
            # Set-up initial prior samples
            ##################
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
            # make the noise blob larger the longer the protein is:
            if self.noise_res_scaling_power > 0:
                trans_0 *= (trans_0.shape[1] / 60.)**self.noise_res_scaling_power

            trans_0 *= self.noise_scale
            ##################

            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

            if self.instance_ot:
                batch['trans_t'] = trans_0
                batch['rotmats_t'] = rotmats_0

                batch = self.permuter(batch)

                trans_0 = batch['trans_t']
                rotmats_0 = batch['rotmats_t']

            # add noise after the order prediction:
            # only really makes sense with instance_ot
            if self.after_ot_noise:
                trans_nm_0_add = _centered_gaussian(num_batch, num_res, self._device)
                trans_0_add = trans_nm_0_add * du.NM_TO_ANG_SCALE
                trans_0_add *= self.noise_res_scaling_power
                trans_0 += trans_0_add
        else:
            raise NotImplementedError("This is not implemented yet")
            trans_0, rotmats_0 = prior_sample
            assert len(rotmats_0.shape) == 4, f"rotmats_0.shape: {rotmats_0.shape}"
            assert len(trans_0.shape) == 3

        if save_path is not None:
            # save trans_0 and rotmats_0
            for i in range(num_batch):
                torch.save(trans_0[i], save_path + f"/sample_{i}/trans_0.pt")
                torch.save(rotmats_0[i], save_path + f"/sample_{i}/rotmats_0.pt")

        # Set up flexibility information if given
        if local_flex is not None and self.classifier_guidance_scale != 0:
            assert local_flex.shape == (num_batch, num_res), f"local_flex.shape: {local_flex.shape} but should be {(num_batch, num_res)}"
            # masking flexibility in batch to -1, so that the model does not use it when inference with CG
            batch['local_flex'] = torch.ones((num_batch, num_res), device=self._device) * (-1.)
        elif local_flex is not None and self.classifier_free_guidance_scale != 0:
            assert local_flex.shape == (num_batch, num_res), f"local_flex.shape: {local_flex.shape} but should be {(num_batch, num_res)}"
            batch['local_flex'] = local_flex
        elif local_flex is None:
            self._log.info("No local flexibility provided, setting to -1 (unconditional sampling).")
            batch['local_flex'] = torch.ones((num_batch, num_res), device=self._device) * (-1.)
        elif self.classifier_guidance_scale == 0 and self.classifier_free_guidance_scale == 0:
            self._log.info("No classifier guidance or conditional flow is enabled, setting local flexibility to -1 (unconditional sampling).")
            batch['local_flex'] = torch.ones((num_batch, num_res), device=self._device) * (-1.)

        # Set-up time
        ts = torch.linspace(
            0., 1.-self._cfg.min_t, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        flex_loss_tensor = []
        flex_loss = None
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step with predicted vector field
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            # Apply classifier guidance
            if self.classifier_guidance_scale != 0 and local_flex is not None:
                # The input to the BackFlip is the noisy state: 
                grads_trans, grads_rot_coords, flex_loss = self.get_backflip_grad(trans_t_2.clone(), rotmats_t_2.clone(), flex_profile=local_flex.float(), res_mask=res_mask)
                if self.length_scaling:
                    # effecitve scale is scaled by the number of residues in the protein to 128 where optimal cg_scale was configured (empirical)
                    effective_scale = self.classifier_guidance_scale * local_flex.shape[1] / 128
                    self.classifier_guidance_scale_schedule = self.cg_scheduler(t_2, ts[-1], effective_scale, schedule=self.cg_schedule)
                else:
                    self.classifier_guidance_scale_schedule = self.cg_scheduler(t_2, ts[-1], self.classifier_guidance_scale, schedule=self.cg_schedule)

                rotmats_t_2 = vf_rots_euler_step(vf=-grads_rot_coords, rotmats=rotmats_t_2, step_size=self.classifier_guidance_scale_schedule)
                trans_t_2 = vf_trans_euler_step(vf=-grads_trans, trans=trans_t_2, step_size=self.classifier_guidance_scale_schedule)

            prot_traj.append((trans_t_2, rotmats_t_2))
            if flex_loss is not None:
                flex_loss_tensor.append(flex_loss.detach().cpu())
            t_1 = t_2

        # Storing flexibility loss tensor
        if len(flex_loss_tensor) > 0:
            flex_loss_tensor = torch.tensor(flex_loss_tensor)
        else:
            flex_loss_tensor = None
        
        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # if given, compute the local flexibility
        if flex_model is not None:
            with torch.no_grad():
                batch['trans_t'] = pred_trans_1
                batch['rotmats_t'] = pred_rotmats_1
                batch['t'] = torch.ones((num_batch, 1), device=self._device).float()
                backflip_input = {
                    'trans_1': pred_trans_1,
                    'rotmats_1': pred_rotmats_1,
                    'res_idx': torch.arange(num_res, device=self._device).unsqueeze(0).expand(num_batch, -1),
                    'res_mask': res_mask,
                }
                flex_result = flex_model(backflip_input)['local_flex'][...,0]
            flex_result_ptr.append(flex_result)
            
        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj, flex_loss_tensor


    def get_backflip_grad(self, trans, rotmats, flex_profile, res_mask=None):
        """
        Computes the gradients of the flexibility profile w.r.t the translation and rotation matrices with BackFlip.
        Args:
            trans: [batch_size, num_res, 3] translation matrices.
            rotmats: [batch_size, num_res, 3, 3] rotation matrices.
            flex_profile: [batch_size, num_res] target flexibility profile.
        Returns:
            grads_trans: [batch_size, num_res, 3] gradients of the flexibility profile wrt translation matrices.
            grads_rot_coords: [batch_size, num_res, 3] gradients of the flexibility profile wrt rotation coordinates.
            flex_loss: scalar tensor representing the flexibility loss.
        """   
        
        num_batch, num_res = trans.shape[:2]
        if res_mask is None:
            res_mask = torch.ones(num_batch, num_res, device=self._device).long()
            
        rot_coords = so3_utils.rotmat_to_rotvec(rotmats)
        with torch.enable_grad():
            trans.requires_grad = True
            rot_coords.requires_grad = True
            # map differentiably from rot_coords to rotmats:
            rotmats_1 = so3_utils.rotvec_to_rotmat(rot_coords)

            backflip_input = {
                'trans_1': trans,
                'rotmats_1': rotmats_1,
                'res_idx': torch.arange(num_res, device=self._device).unsqueeze(0).expand(num_batch, -1),
                'res_mask': res_mask,
            }

            flex_pred = self.backflip(backflip_input)['local_flex'][...,0]
            per_sample_loss = torch.nn.functional.mse_loss(flex_profile.to(self._device), flex_pred, reduction='none')
            flex_loss = per_sample_loss.mean(dim=1).sum() # sum over batch, does not lead to effective scale decay

            grads_trans = torch.autograd.grad(
                outputs=flex_loss,
                inputs=trans,
                retain_graph=True, create_graph=True, allow_unused=False)[0]

            grads_rot_coords = torch.autograd.grad(
                outputs=flex_loss,
                inputs=rot_coords,
                retain_graph=True, create_graph=True, allow_unused=False)[0]
            
        return grads_trans, grads_rot_coords, flex_loss
    

from sklearn.metrics import pairwise_distances
def _ot_permutation(noisy_batch)->torch.Tensor:
    """
    Reorders the noisy trans_0 such that the translational distance between trans_0 and trans_1 is minimal.
    Assumes shapes are [B, N, 3].
    """
    trans_0 = noisy_batch['trans_0']
    trans_1 = noisy_batch['trans_1']
    for b in range(trans_0.shape[0]):
        cost_matrix = pairwise_distances(trans_1[b].numpy(force=True), trans_0[b].numpy(force=True))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        noisy_batch['trans_0'][b] = noisy_batch['trans_0'][b][col_ind]
    return noisy_batch

def vf_rots_euler_step(vf, rotmats, step_size):
    """
    Applies a step of Euler integration to the rotation matrices using the given velocity field.
    """
    rot_coords = so3_utils.rotmat_to_rotvec(rotmats)
    rot_coords += step_size * vf
    rotmats = so3_utils.rotvec_to_rotmat(rot_coords)
    return rotmats

def vf_trans_euler_step(vf, trans, step_size):
    return trans + step_size * vf
