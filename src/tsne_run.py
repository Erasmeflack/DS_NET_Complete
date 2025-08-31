# src/tsne_run.py remains unchanged

import os
import yaml
import torch

from models.dual_snn import DualSiameseNet
from data.load_data import get_mstar_dataloader
from utils.checkpoint import load_checkpoint
from src.analysis.tsne import tsne_one_val_episode

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path="config/5way_1shot.yaml", tag="start", load_best=False):
    cfg = load_cfg(cfg_path)
    device = cfg.get("device", "cuda" if cfg.get("use_gpu", True) and torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    val_dir = os.path.join(cfg["data_dir"], "validate")
    val_loader = get_mstar_dataloader(
        root_dir=val_dir,
        n_way=cfg["n_way"],
        k_shot=cfg["k_shot"],
        q_query=cfg.get("eval_q_query", cfg["q_query"]),
        batch_size=cfg.get("eval_batch_size", 1),
        augment=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.get("prefetch_factor", 2),
    )

    model = DualSiameseNet(
        num_classes=cfg["n_way"],
        agg_method=cfg.get("agg_method", "mean"),
        device=device
    ).to(device)

    if load_best:
        load_checkpoint(model, cfg, optimizer=None, best=True)

    model.eval()
    tsne_one_val_episode(model, val_loader, cfg, tag=tag)

if __name__ == "__main__":
    # Example:
    #   python -m src.tsne_run config/5way_1shot.yaml start False   # untrained
    #   python -m src.tsne_run config/5way_1shot.yaml end True      # best ckpt
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/5way_1shot.yaml"
    tag = sys.argv[2] if len(sys.argv) > 2 else "start"
    load_best = (sys.argv[3].lower() == "true") if len(sys.argv) > 3 else False
    main(cfg_path, tag, load_best)