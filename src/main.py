import argparse
from utils.config import load_yaml_config
from utils.seed import set_seed
from src.train import train
from src.test import test

def main():
    parser = argparse.ArgumentParser(description="Dual Siamese Network Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg.get("random_seed", 42))

    if cfg["is_train"]:
        train(cfg)
    else:
        test(cfg)

if __name__ == "__main__":
    main()