# src/test.py
import os
import csv
import time
import torch
from tqdm import tqdm

from models.dual_snn import DualSiameseNet
from data.load_data import get_mstar_dataloader
from src.trainer.plot_confusion import compute_confusion_matrix
from utils.checkpoint import load_checkpoint
from src.trainer.utils import relation_loss


def _episodic_iter(loader):
    """Yield one episode at a time, regardless of loader batch size."""
    for support_imgs, support_labels, query_imgs, query_labels in loader:
        # If loader returns multi-episode batches [B, N*K, ...], split them.
        if support_imgs.dim() == 6:
            B = support_imgs.size(0)
            for b in range(B):
                yield (
                    support_imgs[b], support_labels[b],
                    query_imgs[b],   query_labels[b]
                )
        else:
            yield (support_imgs, support_labels, query_imgs, query_labels)


def _fine_tune_on_support(model, support_imgs, support_labels, query_imgs, query_labels, cfg):
    """Fine-tune relation/C-Net on the episode's support set (optionally queries too)."""
    if not cfg.get("fine_tune", False):
        return

    # Freeze all first, then unfreeze selected
    for p in model.parameters():
        p.requires_grad = False

    params = []
    if cfg.get("fine_tune_classifier", True) and hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
        params += list(model.classifier.parameters())
    if cfg.get("fine_tune_relation", False) and hasattr(model, "relation_module"):
        for p in model.relation_module.parameters():
            p.requires_grad = True
        params += list(model.relation_module.parameters())

    if not params:
        return

    opt_name = cfg.get("fine_tune_opt", "adam").lower()
    lr = cfg.get("fine_tune_lr", 5e-5)
    OptimCls = getattr(torch.optim, opt_name.capitalize(), torch.optim.Adam)
    optimizer = OptimCls(params, lr=lr)

    model.train()
    epochs = int(cfg.get("fine_tune_epochs", 5))
    for _ in range(epochs):
        optimizer.zero_grad()
        rel_scores, _ = model(query_imgs, support_imgs)
        loss = relation_loss(rel_scores, query_labels, support_labels, cfg["n_way"])
        loss.backward()
        optimizer.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def _apply_ablation(model, mode: str = "full", cfg=None):
    """
    Sets ablation on the model. Supports either:
      - model.set_ablation(use_top, use_mid, use_bottom), or
      - model.use_top / model.use_mid / model.use_bottom boolean attributes.

    Modes:
      "full"        : use all (top/mid/bottom)
      "no_top"      : disable top (similarity)
      "no_mid"      : disable mid (classification branch features into relation)
      "no_bottom"   : disable bottom (dissimilarity)
      "only_top"    : only top
      "only_mid"    : only mid
      "only_bottom" : only bottom
    """
    mode = (mode or "full").lower()

    if mode == "full":
        use_top, use_mid, use_bottom = True, True, True
    elif mode == "no_top":
        use_top, use_mid, use_bottom = False, True, True
    elif mode == "no_mid":
        use_top, use_mid, use_bottom = True, False, True
    elif mode == "no_bottom":
        use_top, use_mid, use_bottom = True, True, False
    elif mode == "only_top":
        use_top, use_mid, use_bottom = True, False, False
    elif mode == "only_mid":
        use_top, use_mid, use_bottom = False, True, False
    elif mode == "only_bottom":
        use_top, use_mid, use_bottom = False, False, True
    else:
        # Fallback to cfg boolean flags if provided
        use_top    = bool(cfg.get("use_top", True)) if cfg else True
        use_mid    = bool(cfg.get("use_mid", True)) if cfg else True
        use_bottom = bool(cfg.get("use_bottom", True)) if cfg else True

    # Try a dedicated API first
    if hasattr(model, "set_ablation") and callable(getattr(model, "set_ablation")):
        model.set_ablation(use_top=use_top, use_mid=use_mid, use_bottom=use_bottom)
    else:
        # Fallback to attributes
        setattr(model, "use_top", use_top)
        setattr(model, "use_mid", use_mid)
        setattr(model, "use_bottom", use_bottom)

    # Keep mode string for reference if the model wants it
    setattr(model, "ablation_mode", mode)


def _evaluate_one_mode(cfg, model, test_loader, class_names, mode_tag: str):
    """
    Runs evaluation for a single ablation mode.
    Returns (best_acc, avg_acc, worst_acc).
    Also writes logs and confusion matrices for this mode.
    """
    device = cfg["device"]

    # CSV logging (per mode)
    log_dir = os.path.join(cfg["logs_dir"], f"exp_{cfg['experiment_name']}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{mode_tag}_test_accuracy.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "accuracy", "timestamp"])

    num_episodes = int(cfg.get("num_episodes", 1000))
    best_acc = 0.0
    worst_acc = 101.0
    total_acc = 0.0
    best_preds, best_labels = [], []
    worst_preds, worst_labels = [], []

    print(f"\n[*] [{mode_tag}] Evaluating over {num_episodes} few-shot episodes...")
    ep_iter = _episodic_iter(test_loader)

    for ep in tqdm(range(num_episodes), desc=f"Episodes ({mode_tag})"):
        try:
            support_imgs, support_labels, query_imgs, query_labels = next(ep_iter)
        except StopIteration:
            ep_iter = _episodic_iter(test_loader)
            support_imgs, support_labels, query_imgs, query_labels = next(ep_iter)

        support_imgs  = support_imgs.to(device, non_blocking=True)
        support_labels= support_labels.to(device, non_blocking=True)
        query_imgs    = query_imgs.to(device, non_blocking=True)
        query_labels  = query_labels.to(device, non_blocking=True)

        # Optional per-episode fine-tune
        if cfg.get("fine_tune", False):
            _fine_tune_on_support(model, support_imgs, support_labels, query_imgs, query_labels, cfg)

        # Evaluate
        model.eval()
        with torch.no_grad():
            relation_scores, _ = model(query_imgs, support_imgs)
            preds = model.predict_from_relations(relation_scores, support_labels)

        preds_np  = preds.detach().cpu().numpy().tolist()
        labels_np = query_labels.detach().cpu().numpy().tolist()

        # Episode accuracy
        correct = sum(1 for p, l in zip(preds_np, labels_np) if p == l)
        ep_acc = 100.0 * correct / max(1, len(labels_np))
        total_acc += ep_acc

        # Log CSV
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, ep_acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

        # Track best and worst episodes
        if ep_acc > best_acc:
            best_acc = ep_acc
            best_preds = preds_np
            best_labels = labels_np
        if ep_acc < worst_acc:
            worst_acc = ep_acc
            worst_preds = preds_np
            worst_labels = labels_np

    avg_acc = total_acc / max(1, num_episodes)

    # Confusion matrices for the best & worst episode (per mode)
    plot_dir = os.path.join(cfg["plot_dir"], f"exp_{cfg['experiment_name']}")
    os.makedirs(plot_dir, exist_ok=True)
    best_path  = os.path.join(plot_dir, f"{mode_tag}_best_confusion.png")
    worst_path = os.path.join(plot_dir, f"{mode_tag}_worst_confusion.png")
    compute_confusion_matrix(best_preds, best_labels, save_path=best_path,  class_names=class_names)
    compute_confusion_matrix(worst_preds, worst_labels, save_path=worst_path, class_names=class_names)

    print(f"[*] [{mode_tag}] Best Episode Accuracy:  {best_acc:.2f}%  (confusion @ {best_path})")
    print(f"[*] [{mode_tag}] Worst Episode Accuracy: {worst_acc:.2f}%  (confusion @ {worst_path})")
    print(f"[*] [{mode_tag}] Average Accuracy over {num_episodes} episodes: {avg_acc:.2f}%")

    return best_acc, avg_acc, worst_acc


def test(cfg):
    device = cfg.get("device", "cuda" if cfg.get("use_gpu", True) and torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    # Episodic test loader
    test_data_dir = os.path.join(cfg["data_dir"], "test")
    test_loader = get_mstar_dataloader(
        root_dir=test_data_dir,
        n_way=cfg["n_way"],
        k_shot=cfg["k_shot"],
        q_query=cfg["q_query"],
        batch_size=cfg.get("eval_batch_size", 1),
        augment=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
    )

    # Class names for confusion matrix
    class_names = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

    # Determine ablation plan
    ablation_modes = cfg.get("ablation_modes")
    if not ablation_modes:
        # Single pass using boolean flags or full (default)
        ablation_modes = [cfg.get("ablation_mode", "full")]

    # Per-mode summary CSV
    log_dir = os.path.join(cfg["logs_dir"], f"exp_{cfg['experiment_name']}")
    os.makedirs(log_dir, exist_ok=True)
    summary_csv = os.path.join(log_dir, "ablation_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerow(["mode", "best_acc", "avg_acc", "worst_acc"])

    summary = []
    for mode in ablation_modes:
        # Fresh model per mode (ensures no contamination from fine-tuning in prior mode)
        model = DualSiameseNet(
            num_classes=cfg["n_way"],
            agg_method=cfg.get("agg_method", "mean"),
            device=device
        ).to(device)

        # Load best/latest weights
        load_checkpoint(model, cfg, optimizer=None, best=cfg.get("best", True))

        # Apply ablation
        _apply_ablation(model, mode, cfg)

        # Evaluate this mode
        best_acc, avg_acc, worst_acc = _evaluate_one_mode(cfg, model, test_loader, class_names, mode)
        summary.append((mode, best_acc, avg_acc, worst_acc))

        # Append to summary CSV
        with open(summary_csv, "a", newline="") as f:
            csv.writer(f).writerow([mode, f"{best_acc:.2f}", f"{avg_acc:.2f}", f"{worst_acc:.2f}"])

    # Print short summary
    print("\n=== Ablation Summary ===")
    for mode, best_acc, avg_acc, worst_acc in summary:
        print(f"{mode:>12} | Best: {best_acc:6.2f}% | Avg: {avg_acc:6.2f}% | Worst: {worst_acc:6.2f}%")
