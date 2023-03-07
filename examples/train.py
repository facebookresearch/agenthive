"""Entry point for RLHive"""
import hydra
from omegaconf import DictConfig

from sac import main as train_sac
from redq import main as train_redq


@hydra.main(config_name="sac_mixed.yaml", config_path="config")
def main(args: DictConfig):
    if args.algo == "sac":
        train_sac(args)
    if args.algo == "redq":
        train_redq(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
