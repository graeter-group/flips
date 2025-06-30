"""DDP inference script."""
import os
import time
import numpy as np
import hydra
import torch
import GPUtil
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from gafl import experiment_utils as eu
from flips.models.flow_module import FlowModule

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Merge configurations and setup directories.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)

        cfg.experiment.checkpointer.dirpath = './'
        # Setting up the configuration.
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self.ckpt_path = ckpt_path
        self._flex_config = cfg.inference.flexibility
        self._interpolant_config = cfg.inference.interpolant
        # Rewrite interpolant config for the inference interpolant passed in the inference_flex config
        cfg.interpolant = self._interpolant_config
        self.min_length = self._flex_config.min_length
        self.max_length = self._flex_config.max_length
        self.length_step = self._flex_config.length_step
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])

        self._output_dir_base = os.path.join(
            self._infer_cfg.output_dir, 
            self._ckpt_name, 
            self._infer_cfg.name
        )

        # Load the flexibility profile and interpolate for different lengths.
        self.flex_profile = np.loadtxt(self._flex_config.flex_profile)
        assert len(self.flex_profile.shape) == 1, "Flexibility profile must be 1D"

        # Load ESM model based on configuration.
        if self._infer_cfg.run_self_consistency:
            self.setup_esm_model()
        self.checkpoint = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)

    def setup_output_directories(self, length):
        self._output_dir = os.path.join(self._output_dir_base, f'length_{length}')
        os.makedirs(self._output_dir, exist_ok=True)
        print(f'Output directory for length {length}: {self._output_dir}')
    
    def interpolate_profile(self, length):
        original_indices = np.linspace(0, 1, len(self.flex_profile))
        new_indices = np.linspace(0, 1, length)
        return np.interp(new_indices, original_indices, self.flex_profile)
    
    def run_sampling_for_each_length(self):
        for length in range(self.min_length, self.max_length + 1, self.length_step):
            # linearly interpolating the input flexibility profile to the desired length
            interpolated_profile = self.interpolate_profile(length)
            self._infer_cfg.min_length = length
            self._infer_cfg.max_length = length
            self.setup_output_directories(length)

            self.loc_save_profile = os.path.join(self._output_dir, f'{length}_profile.txt')
            np.savetxt(self.loc_save_profile, interpolated_profile)
            self._infer_cfg.flexibility.flex_profile = self.loc_save_profile

            self._infer_cfg.samples = OmegaConf.create({
                'num_samples': self._flex_config.num_samples,
                'samples_per_length': self._flex_config.num_samples,
                'batch_size': self._flex_config.batch_size,
                'seq_per_sample': self._infer_cfg.samples.seq_per_sample,
                'min_length': self._infer_cfg.min_length,
                'max_length': self._infer_cfg.max_length,
                'length_step': self.length_step,
                'length_subset': None,
                'overwrite': self._infer_cfg.samples.overwrite,
            })

            # save config
            config_path = os.path.join(self._output_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')
            
            self._samples_cfg = self._infer_cfg.samples
            self._rng = np.random.default_rng(self._infer_cfg.seed)
            # Load model and run sampling
            self.setup_flow_module()
            self.run_sampling()

    def setup_esm_model(self):
        import esm
        Path(self._infer_cfg.pt_hub_dir).mkdir(parents=True, exist_ok=True)
        os.environ['TORCH_HOME'] = self._infer_cfg.pt_hub_dir
        log.info("Load ESM")
        self._folding_model = [esm.pretrained.esmfold_v1().eval()]
        if self._infer_cfg.esmfold_device == 'cpu':
            log.info("Move ESM to CPU")
            self._folding_model[0].esm = self._folding_model[0].esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
            self._folding_model[0] = self._folding_model[0].cpu()
        elif self._infer_cfg.esmfold_device == 'cuda':
            log.info("Move ESM to CUDA")
            self._folding_model[0] = self._folding_model[0].cuda()
        else:
            raise ValueError(f"Unknown device {self._infer_cfg.esmfold_device}")
        log.info("Finished loading ESM")

    def setup_flow_module(self):
        self._flow_module = FlowModule(self._cfg)
        self._flow_module.load_state_dict(self.checkpoint['state_dict'], strict=False)
        self._flow_module.eval()
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir
        self._flow_module._infer_cfg = self._infer_cfg

        if not hasattr(self._infer_cfg, 'max_res_per_esm_batch'):
            self._infer_cfg.max_res_per_esm_batch = 2056
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir
        if self._infer_cfg.run_self_consistency:
            self._flow_module._folding_model = self._folding_model
        else:
            self._flow_module._folding_model = None

    def run_sampling(self):
        devices = GPUtil.getAvailable(order='memory', limit=8)[:self._infer_cfg.num_gpus]
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=devices)
        eval_dataset = eu.LengthDatasetBatch(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer.predict(self._flow_module, dataloaders=dataloader)

@hydra.main(version_base=None, config_path="../configs", config_name="inference_flex")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling_for_each_length()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()