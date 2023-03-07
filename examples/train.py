"""Entry point for RLHive"""
import hydra
from omegaconf import DictConfig

from sac import main as train_sac


@hydra.main(config_name="sac_mixed.yaml", config_path="config")
def main(args: DictConfig):
    if args.algo == "sac":
        train_sac(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
