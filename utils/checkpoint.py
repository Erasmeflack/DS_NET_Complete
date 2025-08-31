# utils/checkpoint.py
import os
import torch


def save_checkpoint(cfg, model, optimizer, epoch, is_best=False):
    ckpt_dir = os.path.join(cfg["ckpt_dir"], f"exp_{cfg['experiment_name']}")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_valid_acc": cfg.get("best_valid_acc", 0.0)
    }

    ckpt_path = os.path.join(ckpt_dir, "model_ckpt.pth")
    torch.save(checkpoint, ckpt_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, "best_model_ckpt.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(model, cfg, optimizer=None, best=False):
    """
    Loads model (and optimizer) state from checkpoint.
    
    Args:
        model: The PyTorch model instance
        cfg (dict): Configuration with paths and device info
        optimizer (optional): Optimizer to load state dict if available
        best (bool): Load best checkpoint if True, else latest

    Returns:
        start_epoch (int): Epoch to resume from
        best_acc (float): Best recorded validation accuracy
    """
    ckpt_dir = os.path.join(cfg["ckpt_dir"], f"exp_{cfg['experiment_name']}")
    filename = 'best_model_ckpt.pth' if best else 'model_ckpt.pth'
    path = os.path.join(ckpt_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"[!] Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=cfg["device"])

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_valid_acc", 0.0)

    print(f"[*] Loaded checkpoint from {path} (Epoch {start_epoch}, Best Acc: {best_acc:.2f}%)")

    return start_epoch, best_acc


def prepare_log_files(cfg):
    """
    Creates:
      - loss.csv with: epoch,sim_loss,dis_loss,cls_loss,rel_loss,uni_loss
      - val.csv  with: epoch,acc,avg_loss
    """
    log_dir = os.path.join(cfg["logs_dir"], f"exp_{cfg['experiment_name']}")
    os.makedirs(log_dir, exist_ok=True)

    paths = {
        "loss_log": os.path.join(log_dir, "loss.csv"),
        "val_log":  os.path.join(log_dir, "val.csv"),
    }

    if not os.path.exists(paths["loss_log"]):
        with open(paths["loss_log"], "w") as f:
            f.write("epoch,sim_loss,dis_loss,cls_loss,rel_loss,uni_loss\n")

    if not os.path.exists(paths["val_log"]):
        with open(paths["val_log"], "w") as f:
            f.write("epoch,acc,avg_loss\n")

    return paths
