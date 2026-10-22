import os

import hydra
from omegaconf import DictConfig

import torch

#Make a more complete trace of errors
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None)
def test_dataset(cfg: DictConfig) -> None:

    print(f"Current working directory : {os.getcwd()}")
    if "dataset_name" in cfg:
        cfg_dataset = cfg[cfg["dataset_name"]]
    else:
        cfg_dataset = cfg.dataset
    dataset = hydra.utils.instantiate(cfg_dataset)
    i = 0
    for data in dataset:
        print(data)
        i += 1
        if i>10:
            break
 
if __name__ == "__main__":
    test_dataset()