# src/train.py

import os
import torch
import random
import numpy as np

from src.trainer.train_loop import train_one_epoch
from src.trainer.validate import validate
from models.dual_snn import DualSiameseNet
from data.load_data import get_mstar_dataloader
from utils.checkpoint import save_checkpoint, prepare_log_files, load_checkpoint


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _make_optimizer(model, cfg):
    name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 0.0)
    params = filter(lambda p: p.requires_grad, model.parameters())

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.get("momentum", 0.9),
            nesterov=True, weight_decay=wd
        )
    elif name == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=lr, momentum=cfg.get("momentum", 0.9),
            weight_decay=wd
        )
    else:
        print(f"[!] Unknown optimizer '{name}', falling back to Adam.")
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def _make_scheduler(optimizer, cfg):
    name = (cfg.get("lr_scheduler") or "").lower()
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.get("lr_factor", 0.5),
            patience=cfg.get("lr_patience", 5),
            min_lr=cfg.get("min_lr", 1e-6),
            verbose=True,
        )
    elif name == "cosineannealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("cosine_tmax", 20),
            eta_min=cfg.get("min_lr", 1e-6),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 20),
            gamma=cfg.get("gamma", 0.1),
        )
    return None


def _ablation_suffix(cfg):
    """
    Build a short suffix describing the ablation setting for logging/ckpt dirs.
    """
    ab = cfg.get("ablation", {})
    rel = int(ab.get("relation_enabled", True))
    use_s = int(ab.get("use_similarity", True))
    use_d = int(ab.get("use_dissimilarity", True))
    use_c = int(ab.get("use_classification", True))
    return f"rel{rel}_S{use_s}_D{use_d}_C{use_c}"


def _apply_ablation_to_cfg(cfg):
    """
    Mutate cfg['experiment_name'] to include ablation suffix so logs/ckpts are isolated per run.
    """
    suffix = _ablation_suffix(cfg)
    base_name = cfg.get("experiment_name", "exp")
    # Avoid double-appending if resuming
    if not base_name.endswith(suffix):
        cfg["experiment_name"] = f"{base_name}_{suffix}"


def train(cfg):
    """
    Main training loop for DS-Net with joint multi-task training, episodic validation, and ablation controls.
    """
    device = cfg.get("device", "cuda" if cfg.get("use_gpu", True) and torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    _set_seed(int(cfg.get("random_seed", 1)))

    # Apply ablation into experiment name so each run is isolated
    _apply_ablation_to_cfg(cfg)

    train_data_dir = os.path.join(cfg["data_dir"], "train")
    val_data_dir   = os.path.join(cfg["data_dir"], "validate")

    # Augment flag handling
    aug_flag = cfg.get("augmentation", True)
    augment = any(aug_flag.values()) if isinstance(aug_flag, dict) else bool(aug_flag)

    # Episodic DataLoaders
    train_loader = get_mstar_dataloader(
        root_dir=train_data_dir,
        n_way=cfg["n_way"],
        k_shot=cfg["k_shot"],
        q_query=cfg["q_query"],
        num_workers=cfg.get("num_workers", 4),
        batch_size=cfg.get("batch_size", 1),
        augment=augment,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = get_mstar_dataloader(
        root_dir=val_data_dir,
        n_way=cfg["n_way"],
        k_shot=cfg["k_shot"],
        q_query=cfg.get("eval_q_query", cfg["q_query"]),  # optionally larger Q for eval
        batch_size=cfg.get("eval_batch_size", 1),
        augment=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
        # prefetch_factor honored in DataLoader ctor if you extend helper; safe to ignore here otherwise
    )

    # --- Ablation flags passed to the model ---
    ab = cfg.get("ablation", {})
    relation_enabled = bool(ab.get("relation_enabled", True))
    use_similarity   = bool(ab.get("use_similarity", True))
    use_dissimilarity= bool(ab.get("use_dissimilarity", True))
    use_classification = bool(ab.get("use_classification", True))

    # Model
    model = DualSiameseNet(
        num_classes=cfg["n_way"],
        agg_method=cfg.get("agg_method", "mean"),
        device=device,
        # ablation controls
        relation_enabled=relation_enabled,
        use_similarity=use_similarity,
        use_dissimilarity=use_dissimilarity,
        use_classification=use_classification,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    # Logs / checkpoints
    log_paths = prepare_log_files(cfg)

    # ---- Resume support ----
    start_epoch = 0
    best_val_acc = 0.0
    if cfg.get("resume", False):
        try:
            start_epoch, best_val_acc = load_checkpoint(model, cfg, optimizer=optimizer, best=False)
            print(f"[*] Resuming training from epoch {start_epoch} (best so far: {best_val_acc:.2f}%)")
        except FileNotFoundError as e:
            print(f"[!] Resume requested, but no checkpoint found: {e}. Starting from scratch.")

    patience_counter = 0
    max_epochs = cfg["epochs"]

    for epoch in range(start_epoch, max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        # One training epoch (joint multi-task loop).
        # train_one_epoch internally respects cfg['episodes_per_epoch'] and ablation flags via model.forward
        train_one_epoch(model, train_loader, optimizer, cfg, epoch, log_paths)

        # Validation (episodic). validate() already supports capping episodes via cfg['val_episodes'].
        val_acc = validate(model, val_loader, cfg, epoch, val_log_path=log_paths["val_log"])

        # LR scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Checkpointing (pass best_val_acc implicitly via "is_best")
        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(cfg, model, optimizer, epoch, is_best=is_best)

        # Early stopping
        if not is_best:
            patience_counter += 1
            if patience_counter >= cfg.get("train_patience", 10):
                print("[!] Early stopping triggered — no improvement.")
                break
        else:
            patience_counter = 0

    print(f"\n[*] Training finished. Best Val Acc: {best_val_acc:.2f}%")
