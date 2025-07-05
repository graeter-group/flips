import os
import GPUtil
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from gafl import experiment_utils as eu
from gafl.models import utils as mu
from gafl.experiment_utils import EpochModelCheckpoint

from flips.models.flow_module import FlowModule
from flips.data.pdb_dataloader import PdbDataModule

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')
class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment

        #Handle missing experiment configs elements
        OmegaConf.set_struct(self._exp_cfg, False)
        if not hasattr(self._exp_cfg.training, 't_bins'):
            self._exp_cfg.training.t_bins = 4

        if not hasattr(self._exp_cfg, 'profiler'):
            self._exp_cfg.profiler = OmegaConf.create()
            self._exp_cfg.profiler.enabled = False

        OmegaConf.set_struct(self._exp_cfg, True)

        self.create_data_module()
        self.create_module()

    def create_module(self):
        self._model: LightningModule = FlowModule(self._cfg)
        if self._exp_cfg.profiler.enabled:
            mu.wrap_model_profiler(self._model, self._exp_cfg.profiler)

    def create_data_module(self):
        self._datamodule: LightningDataModule = PdbDataModule(self._data_cfg)
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 0
        else:
            if self._exp_cfg.use_wandb:
                logger = WandbLogger(
                    **self._exp_cfg.wandb,
                )
            else:
                logger = None
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            print(self._exp_cfg.checkpointer)
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

            if 'checkpointer2' in self._exp_cfg:
                ckpt_dir2 = self._exp_cfg.checkpointer2.dirpath
                os.makedirs(ckpt_dir2, exist_ok=True)
                log.info(f"Additional Checkpoints saved to {ckpt_dir2}")
                callbacks.append(EpochModelCheckpoint(**self._exp_cfg.checkpointer2))
                if 'first_val_epoch' not in self._exp_cfg:
                    callbacks.append(eu.ValidationCallback(self._exp_cfg.checkpointer2.first_epoch))
                elif self._exp_cfg.first_val_epoch > self._exp_cfg.checkpointer2.first_epoch:
                    # Log warning
                    log.warning("First validation epoch is greater than the first checkpoint epoch. Setting first validation epoch to first checkpoint epoch.")
                    callbacks.append(eu.ValidationCallback(self._exp_cfg.checkpointer2.first_epoch))
                else:
                    callbacks.append(eu.ValidationCallback(self._exp_cfg.first_val_epoch))

            if self._exp_cfg.profiler.enabled:
                os.environ["KINETO_LOG_LEVEL"] = "5" # Disable verbose logging of profiler
                callbacks.append(eu.ProfilerCallback())

            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if self._exp_cfg.use_wandb and isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        # NOTE: sorting by order, not by memory. Was causing issues with multigpu training otherwise. 
        devices = GPUtil.getAvailable(order='first', limit = 8)[:self._exp_cfg.num_devices]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            replace_sampler_ddp=False, # in later versions of pytorch lightning, this is called use_distributed_sampler
            enable_progress_bar=self._exp_cfg.use_tqdm,
            enable_model_summary=True,  
            devices=devices,
        )
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    
    # Enable profiling for specified python modules
    if hasattr(cfg.experiment, 'profiler'):
        if cfg.experiment.profiler.enabled:
            mu.wrap_module_profiler(cfg.experiment.profiler)

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
