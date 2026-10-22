import os

import hydra
from omegaconf import DictConfig

import torch

from lightning.pytorch import loggers
from lightning.pytorch import seed_everything

torch.multiprocessing.set_sharing_strategy('file_system')

#Make a more complete trace of errors
os.environ['HYDRA_FULL_ERROR'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# medium or high
torch.set_float32_matmul_precision('medium')

@hydra.main(version_base=None, config_path=".", config_name="test-recipe-base.yaml")
def main(cfg: DictConfig) -> None:

    print(f"Current working directory : {os.getcwd()}")

    data = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model, data)

    trainer.test(model, data)

if __name__ == "__main__":
    main()