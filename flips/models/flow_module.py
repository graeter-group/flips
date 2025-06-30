# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging 
from pytorch_lightning import LightningModule
import shutil
from omegaconf import OmegaConf
import warnings
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning.loggers.wandb import WandbLogger

from gafl.analysis import metrics
from gafl.analysis import utils as au
from gafl.models import utils as mu
from gafl.data import utils as du
from gafl.data import all_atom, so3_utils, residue_constants
from gafl.data.protein import from_pdb_string

from flips.analysis.flex_utils import save_traj
from flips.data.interpolant import Interpolant
from flips.analysis.run_pmpnn_esm import run_self_consistency_batch
from flips.models.flow_model import FlowModel
from flips.models.load_module import load_module

from backflip.deployment.inference_class import BackFlip

# Suppress only the specific PyTorch Lightning user warnings about sync_dist, which are triggered although the world size is 1.
warnings.filterwarnings("ignore", message=".*sync_dist=True.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="opt_einsum.parser")

class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        if not hasattr(cfg.experiment, 'flexibility'):
            OmegaConf.set_struct(cfg.experiment, False)
            cfg.experiment.flexibility = None
            OmegaConf.set_struct(cfg.experiment, True)

        # Flexibility experiment setup:
        self.flexibility_flag = False
        self.flexibility_cfg = cfg.experiment.flexibility if hasattr(cfg.experiment, 'flexibility') else None
        if self.flexibility_cfg is not None:
            if self.flexibility_cfg.flag == True:
                self.flexibility_flag = True

        if self.flexibility_flag:
            # Load BackFlip model
            logging.info(f'Initializing BackFlip')
            flex_model = BackFlip.from_tag('backflip-0.2', device='cuda').model
            # Assign it as a regular attribute to separate it from state dict etc.
            object.__setattr__(self, 'flex_model', flex_model)
            for param in self.flex_model.parameters():
                param.requires_grad = False

        if self.flexibility_flag:
            self.flex_validation_sample_epoch_metrics = []
            self.flex_validation_epoch_samples = []
            self.flex_corr_metrics = []
            self.flex_loss = []
            
        self.flex_mask_prob = self.flexibility_cfg.mask_prob if self.flexibility_flag else None
        self.flex_max_window_size = self.flexibility_cfg.max_window_size if self.flexibility_flag else None
        self.flex_min_window_size = self.flexibility_cfg.min_window_size if self.flexibility_flag else None

        # Set-up vector field prediction model and interpolant
        self.create_model()

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.training_epoch_metrics = []
        self.validation_epoch_metrics = []
        self.validation_sample_epoch_metrics = []
        self.validation_epoch_samples = []
        
        self._time_bins = torch.linspace(0.0, 1.0, 101)
        self._time_histogram = torch.zeros(100, dtype=torch.int64)

        if hasattr(self._exp_cfg, 'warmup_lr'):
            self.warmup_lr = self._exp_cfg.warmup_lr
        else:
            self.warmup_lr = False

        if hasattr(self._exp_cfg, 'warmup_lr_factor'):
            self.warmup_lr_factor = self._exp_cfg.warmup_lr_factor
        else:
            self.warmup_lr_factor = 0.1

        if hasattr(self._exp_cfg, 'reset_optimizer_on_load'):
            self.reset_optimizer_on_load = self._exp_cfg.reset_optimizer_on_load
        else:
            self.reset_optimizer_on_load = False

        self.save_hyperparameters()

    def create_model(self):
        if "module" in self._model_cfg:
            model_module = load_module(self._model_cfg.module)
            self.model = model_module(self._model_cfg)
        else:
            self.model = FlowModel(self._model_cfg)
        self.interpolant = Interpolant(self._interpolant_cfg)

    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        if not self.training_epoch_metrics:
            logging.warning('No training metrics to log')
            self.training_epoch_metrics.clear()
            self._epoch_start_time = time.time()
            return
        
        train_epoch_metrics = pd.concat(self.training_epoch_metrics)
        train_epoch_dict = train_epoch_metrics.mean().to_dict()

        for metric_name,metric_val in train_epoch_dict.items():
            self._log_scalar(
                f'train_epoch/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(train_epoch_metrics),
            )

        self.training_epoch_metrics.clear()

        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self._log_scalar(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def loss_fn(self, noisy_batch: Any, model_output: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        # loss denom may not contain any zeros:
        assert (loss_denom == 0).sum() == 0, 'Loss denom contains zeros'

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        if torch.isnan(se3_vf_loss).any():
            raise ValueError('NaN loss encountered')
        
        # Flexibility auxiliary loss
        flex_aux_loss = None
        if self.flexibility_flag:
            flex_aux_loss_weight = self.flexibility_cfg.aux_loss_weight
            min_time = self.flexibility_cfg.aux_loss_min_time
            assert min_time == 0, 'Flexibility auxiliary loss not implemented for min_time > 0'
        
            gt_local_flex = noisy_batch['local_flex']
            if flex_aux_loss_weight > 0 and torch.any(gt_local_flex != -1.):
                # predict local flexibility using the flex model, such that we can backpropagate through it without changing the flex model weights:
                t_copy = t.clone()
                noisy_batch['t'] = torch.ones_like(noisy_batch['t'])

                backflip_input = {
                'trans_1': noisy_batch['trans_t'],
                'rotmats_1': noisy_batch['rotmats_t'],
                'res_idx': torch.arange(num_res, device=self._device).unsqueeze(0).expand(num_batch, -1),
                'res_mask': noisy_batch['res_mask'],
                }

                pred_local_flex = self.flex_model(backflip_input)['local_flex'][...,0]
                noisy_batch['t'] = t_copy

                # calculate MSE:
                # flex_mask returns a mask tensor with 1s where the local_flex is not -1 else 0
                flex_mask = torch.where(gt_local_flex != -1., 1., 0.)
                loss_mask_copy = torch.where(loss_mask != 0, 1., 0.)
                flex_mask = flex_mask * loss_mask_copy # gives errors for some reason

                flex_aux_loss = torch.sum((gt_local_flex - pred_local_flex)**2 * flex_mask, dim=(-1)) # we dont normalize here
                flex_aux_loss = flex_aux_loss * flex_aux_loss_weight

                if torch.isnan(flex_aux_loss).any():
                    raise ValueError('NaN loss encountered in flexibility auxiliary loss')

                se3_vf_loss += flex_aux_loss

        return_dict = {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
        }
        if flex_aux_loss is not None:
            return_dict['flex_aux_loss'] = flex_aux_loss

        return return_dict


    def model_step(self, noisy_batch: Any):
        model_output = self.model(noisy_batch)
        
        losses = self.loss_fn(noisy_batch, model_output)
        return losses

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        if batch is None:
            return
        
        if dataloader_idx == 0:
            res_mask = batch['res_mask']
            self.interpolant.set_device(res_mask.device)
            num_batch, num_res = res_mask.shape

            samples = self.interpolant.sample(
                num_batch,
                num_res,
                self.model,
            )[0][-1].numpy()

            batch_metrics = []
            for i in range(num_batch):
                # Write out sample to PDB file
                final_pos = samples[i]
                saved_path = au.write_prot_to_pdb(
                    final_pos,
                    os.path.join(
                        self._sample_write_dir,
                        f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                    no_indexing=True
                )
                if isinstance(self.logger, WandbLogger):
                    with open(saved_path, 'r') as f:
                        atom37 = from_pdb_string(f.read()).atom_positions
                    N = atom37.shape[0]
                    backbone = atom37[:,:5,:]
                    colors = np.zeros((N, 5, 3))
                    colors[:,0,:] = np.array([0.0, 0.0, 1.0])
                    colors[:,1,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,2,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,3,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,4,:] = np.array([1.0, 0.0, 0.0])
                    backbone = np.concatenate([backbone, colors*255], axis=-1).reshape(N*5, 6)
                    Ca_atoms = atom37[:,1,:]
                    self.validation_epoch_samples.append(
                        [saved_path, self.global_step, wandb.Molecule(saved_path), wandb.Object3D(backbone), wandb.Object3D(Ca_atoms)]
                    )

                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                ca_idx = residue_constants.atom_order['CA']
                ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

            batch_metrics = pd.DataFrame(batch_metrics)
            self.validation_sample_epoch_metrics.append(batch_metrics)

            if batch_idx == 0:
                self.target_helix_percent = batch['target_helix_percent']
                self.target_strand_percent = batch['target_strand_percent']

            if self.flexibility_flag and num_res == 300 or num_res == 128:
                logging.info(f'Flex validation for len: {num_res}')
                self.flex_validation_step(batch, batch_idx)
                
        if dataloader_idx == 1:
            self.interpolant.set_device(batch['res_mask'].device)
            noisy_batch = self.interpolant.corrupt_batch(batch)
            if self._interpolant_cfg.self_condition and random.random() > 0.5:
                with torch.no_grad():
                    model_sc = self.model(noisy_batch)
                    noisy_batch['trans_sc'] = model_sc['pred_trans']
            batch_losses = self.model_step(noisy_batch)

            batch_metrics = {}
            batch_metrics.update({k: [torch.mean(v).cpu().item()] for k,v in batch_losses.items()})

            # Losses to track. Stratified across t.
            t = torch.squeeze(noisy_batch['t'])
            if self._exp_cfg.training.t_bins > 1:
                for loss_name, loss_dict in batch_losses.items():
                    stratified_losses = mu.t_stratified_loss(
                        t, loss_dict, num_bins=self._exp_cfg.training.t_bins, t_interval=self._interpolant_cfg.t_interval, loss_name=loss_name)
                    batch_metrics.update({k: [v] for k,v in stratified_losses.items()})
        
            batch_metrics = pd.DataFrame(batch_metrics)
            self.validation_epoch_metrics.append(batch_metrics)


    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein", "Backbone", "C-alpha"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        
        if len(self.validation_sample_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.validation_sample_epoch_metrics)

            val_epoch_dict = val_epoch_metrics.mean().to_dict()
            # Calculate deviation of mean helix percent and mean strand percent from dataset
            # this quantity is actually the mean of the mean of the residue-level helix and strand percent! (which is fine, otherwise small proteins would have less weight)
            helix_deviation = val_epoch_metrics['helix_percent'].mean() - self.target_helix_percent
            strand_deviation = val_epoch_metrics['strand_percent'].mean() - self.target_strand_percent
            sec_deviation = abs(helix_deviation) + abs(strand_deviation)
            self._log_scalar(
                'valid/sec_deviation',
                sec_deviation,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics)
            )

            for metric_name,metric_val in val_epoch_dict.items():
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            self.validation_sample_epoch_metrics.clear()

        if self.flexibility_flag and len(self.flex_validation_epoch_samples) > 0:
            self.flex_on_validation_epoch_end()

        
        if len(self.validation_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.validation_epoch_metrics)

            val_epoch_dict = val_epoch_metrics.mean().to_dict()

            for metric_name,metric_val in val_epoch_dict.items():
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
        

            self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist is None:
            sync_dist = self.trainer.world_size > 1
        if rank_zero_only is None:
            rank_zero_only = not sync_dist
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def mask_flexibility(self, batch: Any):
        '''
        Masks local_flexibility values in the batch for the training
        Masking strategy:
        mask_encode:
            0 -> full mask, local_flex is ignored
            1 -> full UNmask, local_flex is used as is
            # num_res / 2 was changed
            2 -> UNmask a random window of residues (size: num_res/2) in each instance
        This happens based on mask_prob: [p0, p1, p2] where p0 is the probability of hiding the flexibility completely, p1 is the probability of using the flexibility as is, and p2 is the probability of unmasking a random window
        If self.flex_mask_prob is None, then the flexibility is hidden completely
        '''
        if 'local_flex' in batch.keys():
            local_flex = batch['local_flex']
            if not torch.is_tensor(local_flex):
                local_flex = torch.tensor(batch['local_flex'], device=batch['res_mask'].device)
            
            # shape of local_flex: [batch_dims, num_res]
            batch_dims = local_flex.shape[:-1]
            num_res = local_flex.shape[-1]
            
            mask_tensor = torch.zeros_like(local_flex)
            
            if self.flex_mask_prob is not None:
                # decide how to mask: self.flex_mask_prob is passed as a list of weights for each option
                mask_encode = np.random.choice([0, 1, 2], p=self.flex_mask_prob)
                if mask_encode == 0:
                    pass
                elif mask_encode == 1:
                    mask_tensor = torch.ones_like(local_flex)
                else:
                    # Randomly choose a window position between min_windows_size and max_window_size and only unmask this window
                    min_window_size = int(self.flex_min_window_size * num_res)
                    max_window_size = int(self.flex_max_window_size * num_res)

                    # choose center and window size:
                    pos = torch.randint(low=0, high=num_res-1, size=batch_dims, device=local_flex.device)
                    size = torch.randint(low=min_window_size, high=max_window_size, size=batch_dims, device=local_flex.device)

                    # clamp the windows:
                    # shape: [batch_dims, num_res]
                    start_pos = torch.clamp(pos - size, min=0).int().cpu()
                    end_pos = torch.clamp(pos + size, max=num_res).int().cpu()

                    # Generate a mask for the specified range
                    # mask tensor has shape [batch_dims, num_res] with all zeros, set them to one at the window positions:
                    for i in range(len(batch_dims)):
                        mask_tensor[i, start_pos[i]:end_pos[i]] = 1
            else:
                # if probs are not passed, flexibility is not used -> set to -1
                pass
        else:
            mask_tensor = torch.zeros_like(batch['res_mask'])
            local_flex = torch.full((batch['res_mask'].shape[-1],), -1, device=batch['res_mask'].device)

        local_flex = torch.where(mask_tensor == 1, local_flex, -1)
        return local_flex
    
    def training_step(self, batch: Any, stage: int):

        batchsize = batch['res_mask'].shape[0]
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        
        if 'local_flex' not in batch.keys():
            print(f'Local flex is not in keys, setting to -1')
            local_flex_masked = self.mask_flexibility(batch)
        else:
            batch['local_flex'] = copy.deepcopy(batch['local_flex'])
            local_flex_masked = self.mask_flexibility(batch)
        
        batch['local_flex'] = local_flex_masked

        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        batch_metrics = {}
        batch_metrics.update({k: [v.cpu().item()] for k,v in total_losses.items()})

        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        
        if self._exp_cfg.training.t_bins > 1:
            for loss_name, loss_dict in batch_losses.items():
                stratified_losses = mu.t_stratified_loss(
                    t, loss_dict, num_bins=self._exp_cfg.training.t_bins, t_interval=self._interpolant_cfg.t_interval, loss_name=loss_name)
                batch_metrics.update({k: [v] for k,v in stratified_losses.items()})

                for k,v in stratified_losses.items():
                    self._log_scalar(
                        f"train/{k}", v, prog_bar=False, batch_size=num_batch)

                batch_metrics = pd.DataFrame(batch_metrics)
        self.training_epoch_metrics.append(batch_metrics)
        
        # Training throughput
        self._log_scalar(
            "train/length", float(batch['res_mask'].shape[1]), prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", float(num_batch), prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = (
            total_losses[self._exp_cfg.training.loss]
            +  total_losses['auxiliary_loss']
        ) # This double counts the auxiliary loss
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        
        if self._exp_cfg.use_wandb:
            wandb_logs = {}
            time_indices = torch.bucketize(noisy_batch["t"].detach().cpu(), self._time_bins)
            time_indices_unique, counts = torch.unique(time_indices-1, return_counts=True)
            time_indices_unique[time_indices_unique < 0] = 0
            self._time_histogram[time_indices_unique] += counts
            wandb_logs["sampled_time_cumulative"] = wandb.Histogram(
                np_histogram=((
                    (self._time_histogram/self._time_histogram.sum()).numpy(),
                    self._time_bins.numpy()
                ))
            )
            time_histogram_step = torch.zeros(self._time_histogram.shape)
            time_histogram_step[time_indices_unique] += counts
            wandb_logs["sampled_time_per_step"] = wandb.Histogram(
                np_histogram=((
                    (time_histogram_step/time_histogram_step.sum()).numpy(),
                    self._time_bins.numpy()
                ))
            )
            self.logger.experiment.log(wandb_logs)


        return train_loss

    def configure_optimizers(self):
        if not self.warmup_lr:
            return torch.optim.AdamW(
                params=self.model.parameters(),
                **self._exp_cfg.optimizer
            )
        else:
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                **self._exp_cfg.optimizer
            )
            if self.warmup_lr_factor == 0:
                return optimizer
            
            # train the first epoch with a smaller learning rate:
            small_lr = self.warmup_lr_factor * self._exp_cfg.optimizer.lr
            this_epoch = self.trainer.current_epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda
                    epoch: 1.0 if epoch >= this_epoch else small_lr
            )
            return [optimizer], [scheduler]
    
    def on_load_checkpoint(self, *args, **kwargs):
        output = super().on_load_checkpoint(*args, **kwargs)
        if self.reset_optimizer_on_load:
            self.configure_optimizers()
        return output


    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length)
        all_sample_ids = batch['sample_id'][0].cpu().numpy()

        if hasattr(self._infer_cfg, 'flexibility'):
            profile_path = self._infer_cfg.flexibility.flex_profile
            if profile_path is None:
                logging.warning('Flexibility profile path is not set, setting local_flex_target to None.')
                local_flex_target = None
            local_flex_target = torch.tensor(np.loadtxt((profile_path)), device=device)
            # repeat n_instance_per_batch times
            local_flex_target = local_flex_target
        else:
            local_flex_target = None

        batch_size = self._infer_cfg.samples.batch_size
        sample_ids_batches = np.split(
            all_sample_ids,
            np.arange(batch_size, len(all_sample_ids), batch_size)
        )

        for sample_ids in sample_ids_batches:
            length_dir = os.path.join(self._output_dir, f'length_{sample_length}')

            for i, sample_id in enumerate(sample_ids):
                sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
                os.makedirs(sample_dir, exist_ok=True)

            flex_result_ptr = [] # use that python lists behave like pointers to pass the result without modifying the function return
            local_flex_target_ = local_flex_target.repeat(len(sample_ids), 1) if local_flex_target is not None else None
            if hasattr(self._infer_cfg, 'flexibility'):
                local_flex_condition = local_flex_target_ if self._infer_cfg.flexibility.pass_flex_condition else None
            else:
                local_flex_condition = None
            
            # logging.info(f'Started sampling for {len(sample_ids)} samples with length {sample_length}')
            atom37_traj, model_traj, _, flex_loss_tensor = interpolant.sample(
                len(sample_ids), sample_length, self.model, save_path=length_dir,
                local_flex=local_flex_condition,
                flex_model=self.flex_model,
                flex_result_ptr=flex_result_ptr,
            )
            if flex_loss_tensor != None:
                flex_loss_tensor = du.to_numpy(flex_loss_tensor)
                self.flex_loss.append(flex_loss_tensor)

            predicted_local_flex = None if len(flex_result_ptr) == 0 or flex_result_ptr is None else flex_result_ptr[0].cpu().numpy()

            local_flex_target_ = local_flex_target_.cpu().numpy() if local_flex_target is not None else None

            atom37_traj = du.to_numpy(torch.stack(atom37_traj, dim=1))
            model_traj = du.to_numpy(torch.stack(model_traj, dim=1))

            for i, sample_id in enumerate(sample_ids):
                sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
                paths = save_traj(
                    atom37_traj[i, -1],
                    np.flip(atom37_traj[i], axis=0),
                    np.flip(model_traj[i], axis=0),
                    du.to_numpy(diffuse_mask)[0],
                    output_dir=sample_dir,
                    local_flex=local_flex_target_[i] if local_flex_target_ is not None else None,
                    predicted_local_flex=predicted_local_flex[i] if predicted_local_flex is not None else None,
                    write_traj=True if self._infer_cfg.write_trajectory else False,
                )
                if self._infer_cfg.run_self_consistency:
                    sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                    os.makedirs(sc_output_dir, exist_ok=True)
                    pdb_path = paths["sample_path"]
                    shutil.copy(pdb_path, os.path.join(sc_output_dir, os.path.basename(pdb_path)))
            
            # Run self-consistency for all predicted samples if no flexibility in the config or backflip screening is disabled.
            if not hasattr(self._infer_cfg, 'flexibility') or not self._infer_cfg.flexibility.backflip_screening:
                if self._infer_cfg.run_self_consistency:
                    run_self_consistency_batch(
                        self._infer_cfg.pmpnn_dir,
                        self._infer_cfg.samples.seq_per_sample,
                        self._folding_model[0],
                        length_dir,
                        sample_ids,
                        motif_mask=None,
                        calc_non_coil_rmsd=self._infer_cfg.calc_non_coil_rmsd,
                        max_res_per_esm_batch=self._infer_cfg.max_res_per_esm_batch,
                    )
            else:
                # calculate the correlations:
                if self._infer_cfg.flexibility.backflip_screening:
                    assert local_flex_target_ is not None
                    assert predicted_local_flex is not None
                    for i, sample_id in enumerate(sample_ids):
                        target_flex = local_flex_target_[i]
                        predicted_flex = predicted_local_flex[i][target_flex >= 0]
                        target_flex = target_flex[target_flex >= 0]

                        assert len(target_flex)>0, 'No flexibility conditions found'

                        corr_coeff = np.corrcoef(target_flex, predicted_flex)[0, 1]
                        self.flex_correlations.append(corr_coeff)
                        self.flex_rmses.append(np.sqrt(np.mean((target_flex - predicted_flex)**2)))
                        self.flex_maes.append(np.mean(np.abs(target_flex - predicted_flex)))
                        self.length_dirs.append(length_dir)
                        self.sample_ids.append(sample_id)                    

    def to(self, device):
        if self.flexibility_flag:
            self.flex_model = self.flex_model.to(device)
        return super().to(device)
    

    def get_val_flex_targets(self, res_mask):
        '''
        Returns a list of flexibility targets for validation
        with slice regions computed as fractions of length 128 (benchmarked on this len)
        '''
        FLEX_TARGETS = []
        L = res_mask.shape[1]
        local_flex_base = torch.ones_like(res_mask) * 0.5

        def pct(p): return int(p * L / 128)

        # 1 window
        local_flex = local_flex_base.clone()
        local_flex[:, pct(10):pct(15)] = 1.3
        local_flex[:, pct(45):pct(55)] = 1.7
        local_flex[:, pct(80):pct(85)] = 1.5
        local_flex[:, pct(100):pct(105)] = 1.2
        local_flex[:, pct(118):pct(128)] = 0.8
        FLEX_TARGETS.append(local_flex)

        # 2 windows
        local_flex = local_flex_base.clone()
        local_flex[:, pct(40):pct(50)] = 1.
        local_flex[:, pct(80):pct(90)] = 2.5
        FLEX_TARGETS.append(local_flex)

        # 3 windows
        local_flex = local_flex_base.clone()
        local_flex[:, pct(30):pct(40)] = 1.
        local_flex[:, pct(60):pct(70)] = 1.5
        local_flex[:, pct(100):pct(110)] = 2.
        FLEX_TARGETS.append(local_flex)

        # complex
        local_flex = local_flex_base.clone()
        local_flex[:, pct(0):pct(5)] = 1.5
        local_flex[:, pct(20):pct(26)] = 1.0
        local_flex[:, pct(55):pct(60)] = 2.0
        local_flex[:, pct(73):pct(77)] = 1.4
        local_flex[:, pct(95):pct(105)] = 2.2
        FLEX_TARGETS.append(local_flex)

        return FLEX_TARGETS


    def flex_validation_step(self, batch, batch_idx):
        assert self.flexibility_flag, 'Flexibility flag is not set but trying to run flex validation step'

        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape

        # sample again with different flexibility targets:
        # create some flexibility targets:
        FLEX_TARGETS = self.get_val_flex_targets(res_mask)

        calc_correlation = [] # for what entries of FLEX_TARGETS to calculate correlation (makes only sense if whole range is specified, not for windows) where the rest is unspecified (set to -1) )
        if 'local_flex' in batch.keys():
            this_flex = batch['local_flex'].clone()
            FLEX_TARGETS = [this_flex] + FLEX_TARGETS

        calc_correlation = list(range(len(FLEX_TARGETS)))

        for flex_idx, local_flex_target in enumerate(FLEX_TARGETS):
            flex_result_ptr = [] # result will be written here
            batch_metrics = []
            corr_metric = []
            samples = self.interpolant.sample(
                num_batch,
                num_res,
                self.model,
                local_flex=local_flex_target,
                flex_model=self.flex_model,
                flex_result_ptr=flex_result_ptr,
            )[0][-1].numpy()

            predicted_local_flex_batch = flex_result_ptr[0].cpu().numpy()
            local_flex_target = local_flex_target.cpu().numpy()
            for i in range(num_batch):
                # push a plot of predicted and target local flexibilities
                target_flex = local_flex_target[i]
                predicted_flex = predicted_local_flex_batch[i]

                if flex_idx in calc_correlation:
                    corr_coeff = np.corrcoef(target_flex, predicted_flex)[0, 1]
                    corr_metric.append({'Flex Correlation': corr_coeff})

                plotpath = os.path.join(
                    self._sample_write_dir,
                    f'flex{flex_idx}_sample_{i}_idx_{batch_idx}_len_{num_res}.png'
                )
                fig, ax = plt.subplots()
                ax.plot(np.arange(num_res), target_flex, label='Target')
                ax.plot(np.arange(num_res), predicted_flex, label='Sample (Pred.)')
                ax.set_xlabel('Residue Index')
                ax.set_ylabel('Local Flexibility')
                fig.legend()
                fig.savefig(plotpath)
                plt.close(fig)

                # store pdbs:
                final_pos = samples[i]
                saved_path = au.write_prot_to_pdb(
                    final_pos,
                    os.path.join(
                        self._sample_write_dir,
                        f'flex{flex_idx}_sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                    no_indexing=True
                )

                if isinstance(self.logger, WandbLogger):
                    with open(saved_path, 'r') as f:
                        atom37 = from_pdb_string(f.read()).atom_positions
                    N = atom37.shape[0]
                    backbone = atom37[:,:5,:]
                    colors = np.zeros((N, 5, 3))
                    colors[:,0,:] = np.array([0.0, 0.0, 1.0])
                    colors[:,1,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,2,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,3,:] = np.array([0.25, 0.75, 0.75])
                    colors[:,4,:] = np.array([1.0, 0.0, 0.0])
                    backbone = np.concatenate([backbone, colors*255], axis=-1).reshape(N*5, 6)
                    Ca_atoms = atom37[:,1,:]
                    self.flex_validation_epoch_samples.append(
                        # [saved_path, self.global_step, wandb.Molecule(saved_path)]#, compare_flex_plot]
                        [saved_path, wandb.Molecule(saved_path), wandb.Image(plotpath)]
                    )

                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                ca_idx = residue_constants.atom_order['CA']
                ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

            batch_metrics = pd.DataFrame(batch_metrics)
            self.flex_validation_sample_epoch_metrics.append(batch_metrics)
            if len(corr_metric) > 0:
                corr_metric = pd.DataFrame(corr_metric)
                self.flex_corr_metrics.append(corr_metric)


    def flex_on_validation_epoch_end(self):
        if len(self.flex_validation_epoch_samples) > 0:
            self.logger.log_table(
                key='flex_valid/samples',
                columns=["sample_path", "Protein", "Flexibility"],
                data=self.flex_validation_epoch_samples)
            self.flex_validation_epoch_samples.clear()
        
        if len(self.flex_validation_sample_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.flex_validation_sample_epoch_metrics)

            val_epoch_dict = val_epoch_metrics.mean().to_dict()

            if len(self.flex_corr_metrics) > 0:
                corr_metrics = pd.concat(self.flex_corr_metrics)
                corr_dict = corr_metrics.mean().to_dict()
                val_epoch_dict.update(corr_dict)

            # Calculate deviation of mean helix percent and mean strand percent from dataset
            # this quantity is actually the mean of the mean of the residue-level helix and strand percent! (which is fine, otherwise small proteins would have less weight)
            helix_deviation = val_epoch_metrics['helix_percent'].mean() - self.target_helix_percent
            strand_deviation = val_epoch_metrics['strand_percent'].mean() - self.target_strand_percent
            sec_deviation = abs(helix_deviation) + abs(strand_deviation)

            self._log_scalar(
                'flex_valid/sec_deviation',
                sec_deviation,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics)
            )
            
            # using flex_corr_mean for logging now
            flex_corr_mean = corr_metrics['Flex Correlation'].mean()
            flex_corr_mean = torch.tensor(flex_corr_mean, device=sec_deviation.device, dtype=sec_deviation.dtype)

            self._log_scalar(
                'flex_valid/flex_corr_mean',
                flex_corr_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics)
            )

            for metric_name,metric_val in val_epoch_dict.items():
                self._log_scalar(
                    f'flex_valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            self.flex_validation_sample_epoch_metrics.clear()
            if len(self.flex_corr_metrics) > 0:
                self.flex_corr_metrics.clear()


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Call the base to get the state dict and then delete flex_model related keys
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Remove flex_model related keys
        for key in list(state.keys()):
            if key.startswith('flex_model.'):
                del state[key]
        return state

    def load_state_dict(self, state_dict, strict=False):
        # Remove flex_model keys if present in the state_dict to avoid errors
        for key in list(state_dict.keys()):
            if key.startswith('flex_model.'):
                del state_dict[key]
        # Load state dict normally
        super().load_state_dict(state_dict, strict=strict)

    def on_predict_epoch_start(self, *args, **kwargs):
        if hasattr(self._infer_cfg, 'flexibility'):
            if self._infer_cfg.flexibility.backflip_screening:
                logging.info(f'Loading BackFlip from tag "backflip-0.2"')
                self.flex_model = BackFlip.from_tag('backflip-0.2', device='cuda').model
                for param in self.flex_model.parameters():
                    param.requires_grad = False
                self.flexibility_flag = True
            else:
                logging.info(f'BackFlip screening is disabled, inference without BackFlip model.')
                self.flex_model = None
        else:
            self.flex_model = None

        if hasattr(self._infer_cfg, 'flexibility'):
            self.flex_correlations = []
            self.flex_rmses = []
            self.flex_maes = []
            self.length_dirs = []
            self.sample_ids = []


    def on_predict_epoch_end(self, *args, **kwargs):
        if hasattr(self._infer_cfg, 'flexibility'):
            if self._infer_cfg.flexibility.backflip_screening:
                n_top_score = self._infer_cfg.flexibility.num_top_samples
                # sort the correlations and apply the same permutation to the length_dirs and sample_ids
                self.flex_correlations = np.array(self.flex_correlations)
                self.flex_rmses = np.array(self.flex_rmses)
                self.flex_maes = np.array(self.flex_maes)
                self.flex_loss = np.array(self.flex_loss)

                #NOTE: originally during training scored with RMSE:
                # scores = self.flex_correlations * self._infer_cfg.flexibility.correlation_weight - self.flex_rmses * self._infer_cfg.flexibility.rmse_weight
                scores = self.flex_correlations * self._infer_cfg.flexibility.correlation_weight - self.flex_maes * self._infer_cfg.flexibility.mae_weight
                
                # sort descending and get the top n_top_score samples:
                top_score_indices = np.argsort(scores)[::-1][:n_top_score]
                top_correlations = self.flex_correlations[top_score_indices]
                top_rmses = self.flex_rmses[top_score_indices]
                top_maes = self.flex_maes[top_score_indices]
                top_length_dirs = [self.length_dirs[i] for i in top_score_indices]
                top_sample_ids = [self.sample_ids[i] for i in top_score_indices]

                # for now, assume we have only one length:
                assert len(list(set(top_length_dirs))) == 1
                length_dir = top_length_dirs[0]

                # now copy the top sample dirs to a 'top_flex' directory:
                top_dir = os.path.join(self._output_dir, 'top_flex')
                top_len_dir = os.path.join(top_dir, Path(length_dir).name)
                self._output_dir = top_dir
                os.makedirs(top_dir, exist_ok=True)
                os.makedirs(top_len_dir, exist_ok=True)
                for i, sample_id in enumerate(top_sample_ids):
                    sample_dir = os.path.join(length_dir, f'sample_{sample_id}')
                    top_sample_dir = os.path.join(top_len_dir, f'sample_{sample_id}')
                    shutil.copytree(sample_dir, top_sample_dir)

                # log the correlations and flex maes:
                df = pd.DataFrame({
                    'Sample ID': top_sample_ids,
                    'Flex Correlation': top_correlations,
                    'Flex MAE': top_maes,
                    'Score': scores[top_score_indices]
                })
                
                logging.info(f'Flexibility Correlations and MAEs for top {n_top_score} samples:\n{str(df)}')

                with open(os.path.join(top_dir, 'flex_correlations.csv'), 'w') as f:
                    df.to_csv(f, index=False)
                
                with open(os.path.join(top_dir, 'flex_correlations.txt'), 'w') as f:
                    f.write(str(df))
                
                if len(self.flex_loss) > 0:
                    df_loss = pd.DataFrame(self.flex_loss.T, columns=[f'batch_{i}' for i in range(self.flex_loss.shape[0])])
                    df_loss.to_csv(os.path.join(top_dir, 'flex_loss.csv'), index=False)
            
                if self._infer_cfg.run_self_consistency:
                    run_self_consistency_batch(
                        self._infer_cfg.pmpnn_dir,
                        self._infer_cfg.samples.seq_per_sample,
                        self._folding_model[0],
                        top_len_dir,
                        top_sample_ids,
                        weights=self._infer_cfg.pmpnn_weights,
                        motif_mask=None,
                        calc_non_coil_rmsd=self._infer_cfg.calc_non_coil_rmsd,
                        max_res_per_esm_batch=self._infer_cfg.max_res_per_esm_batch,
                    )