import hydra
import sys, os
sys.path.append(os.getcwd())  # add current folder to path

from omegaconf import DictConfig



# For complete error traces
os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base=None, config_path=".", config_name="test-recipe-union-load.yaml")
def main(cfg: DictConfig) -> None:

    data = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model, data)

    trainer.test(model, data)



if __name__ == "__main__":

    main()