# src/test.py
import os
import csv
import time
import torch
from tqdm import tqdm

from models.dual_snn import DualSiameseNet
from data.load_data import get_mstar_dataloader
from src.trainer.plot_confusion import plot_confusion_from_preds
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
    """
    Fine-tune on the current episode depending on which modules are selected:
      - classifier only: classification CE on C-Net logits (query -> logits via forward_pairwise(q,q))
      - relation only:   relation CE on relation scores (query vs support)
      - both:            weighted sum of relation CE + classification CE
    """
    if not cfg.get("fine_tune", False):
        return

    device = cfg["device"]
    model.train()

    # 1) Freeze everything, then unfreeze chosen modules
    for p in model.parameters():
        p.requires_grad = False

    ft_cls  = bool(cfg.get("fine_tune_classifier", True))
    ft_rel  = bool(cfg.get("fine_tune_relation", False))

    params = []
    used_modules = []

    if ft_cls and hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
        params += list(model.classifier.parameters())
        used_modules.append("classifier")

    if ft_rel and hasattr(model, "relation_module"):
        for p in model.relation_module.parameters():
            p.requires_grad = True
        params += list(model.relation_module.parameters())
        used_modules.append("relation_module")

    if not params:
        print("[fine-tune] No modules selected; skipping fine-tuning.")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return

    # 2) Optimizer and schedule
    opt_name = cfg.get("fine_tune_opt", "adam").lower()
    lr = float(cfg.get("fine_tune_lr", 5e-5))
    epochs = int(cfg.get("fine_tune_epochs", 5))

    OptimCls = getattr(torch.optim, opt_name.capitalize(), torch.optim.Adam)
    optimizer = OptimCls(params, lr=lr)

    lam_rel = float(cfg.get("ft_lambda_rel", 1.0))
    lam_cls = float(cfg.get("ft_lambda_cls", 1.0))

    # 3) Print plan
    loss_plan = []
    if ft_rel: loss_plan.append("relationCE")
    if ft_cls: loss_plan.append("classCE")
    print(f"[fine-tune] Modules: {', '.join(used_modules)} | losses: {', '.join(loss_plan)} | "
          f"optimizer={OptimCls.__name__} | lr={lr} | epochs={epochs}")

    # Ensure tensors are on device and shaped right
    support_imgs   = support_imgs.to(device)
    support_labels = support_labels.view(-1).to(device)
    query_imgs     = query_imgs.to(device)
    query_labels   = query_labels.view(-1).to(device)

    # 4) Run a few epochs on the episode
    for ep in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        # Relation loss branch (if enabled)
        if ft_rel:
            rel_scores, _ = model(query_imgs, support_imgs)  # (Q,S,1)
            rel_ce = relation_loss(rel_scores, query_labels, support_labels, cfg["n_way"])
            total_loss = total_loss + lam_rel * rel_ce

        # Classification loss branch (if enabled)
        if ft_cls:
            # Build logits for all queries using C-Net path (classifier head)
            cls_logits_list = []
            for q_idx in range(query_imgs.size(0)):
                q = query_imgs[q_idx].unsqueeze(0)
                # forward_pairwise returns (sim_score, dis_score, class_logits)
                _, _, logits = model.forward_pairwise(q, q)
                cls_logits_list.append(logits)
            if cls_logits_list:
                cls_logits = torch.cat(cls_logits_list, dim=0)  # [Q, n_way]
                cls_ce = torch.nn.functional.cross_entropy(cls_logits, query_labels)
                total_loss = total_loss + lam_cls * cls_ce
            else:
                cls_ce = torch.tensor(0.0, device=device)

        # Safety: if for some reason total_loss has no grad (shouldn't happen), skip
        if not hasattr(total_loss, "backward"):
            print("[fine-tune] Warning: loss has no backward; skipping this FT step.")
        else:
            total_loss.backward()
            optimizer.step()

    # 5) Back to eval, keep grads off
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

    print(f"[ablation] mode={mode} | use_top={use_top}, use_mid={use_mid}, use_bottom={use_bottom}")


def _evaluate_one_mode(cfg, model, test_loader, class_names, mode_tag: str):
    """
    Evaluate one ablation mode over cfg['num_episodes'] episodes.
    One batch = one episode. Logs per-episode accuracy to CSV.
    """
    device = cfg["device"]
    num_episodes = int(cfg.get("num_episodes", 1000))
    base_log_dir = os.path.join(cfg.get("logs_dir", "./logs"), f"{cfg['experiment_name']}")
    os.makedirs(base_log_dir, exist_ok=True)

    # Per-episode accuracy CSV for this mode
    iter_csv = os.path.join(base_log_dir, f"test_iter_{mode_tag}.csv")
    
    # Initialize the iterator here
    test_loader_iter = iter(test_loader)
    
    # Open CSV once and use a writer object
    with open(iter_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "accuracy", "num_queries", "correct", "timestamp"])
        
        plot_dir = os.path.join(cfg.get("plot_dir", "./plots"), f"{cfg['experiment_name']}")
        os.makedirs(plot_dir, exist_ok=True)

        best_acc, worst_acc = -1.0, 101.0
        sum_acc = 0.0
        processed_eps = 0
        actual_episode_count = 0  # Track actual episodes processed
        best_preds, best_labels = [], []
        worst_preds, worst_labels = [], []

        # Use tqdm for progress tracking
        pbar = tqdm(total=num_episodes, desc=f"Evaluating {mode_tag}")
        
        for episode_idx in range(num_episodes):
            # --- fetch exactly one episode ---
            try:
                support_imgs, support_labels, query_imgs, query_labels = next(test_loader_iter)
            except StopIteration:
                # Reset iterator when exhausted
                test_loader_iter = iter(test_loader)
                try:
                    support_imgs, support_labels, query_imgs, query_labels = next(test_loader_iter)
                except StopIteration:
                    # Truly empty loader - break out of the loop
                    pbar.set_description(f"{mode_tag} - Loader exhausted")
                    break

            # Move to device & normalize shape
            support_imgs = support_imgs.to(device)
            support_labels = support_labels.view(-1).to(device)
            query_imgs = query_imgs.to(device)
            query_labels = query_labels.view(-1).to(device)

            if query_labels.numel() == 0:
                # skip empty episode, don't count towards average
                pbar.update(1)
                continue

            if cfg.get("fine_tune", False):
                print(f"[fine-tune:{mode_tag}] Starting per-episode fine-tune...")
                _fine_tune_on_support(model, support_imgs, support_labels, query_imgs, query_labels, cfg)

            # Inference for this episode
            with torch.no_grad():
                relation_scores, _ = model(query_imgs, support_imgs)
                if relation_scores is None or relation_scores.numel() == 0:
                    # log as 0% accuracy if something odd occurs
                    ep_acc = 0.0
                    writer.writerow([actual_episode_count, f"{ep_acc:.2f}", 0, 0, 
                                   time.strftime("%Y-%m-%d %H:%M:%S")])
                    pbar.update(1)
                    actual_episode_count += 1
                    continue

                preds = model.predict_from_relations(relation_scores, support_labels)  # [Q]

            # Compute episode acc
            preds_list = preds.detach().cpu().tolist()
            labels_list = query_labels.detach().cpu().tolist()

            correct = sum(int(p == l) for p, l in zip(preds_list, labels_list))
            ep_acc = 100.0 * correct / max(1, len(labels_list))
            sum_acc += ep_acc
            processed_eps += 1

            # --- per-episode logging ---
            writer.writerow([actual_episode_count, f"{ep_acc:.2f}", len(labels_list), correct, 
                           time.strftime("%Y-%m-%d %H:%M:%S")])

            # Track best/worst for confusion plots
            if ep_acc > best_acc:
                best_acc = ep_acc
                best_preds, best_labels = preds_list.copy(), labels_list.copy()

            if ep_acc < worst_acc:
                worst_acc = ep_acc
                worst_preds, worst_labels = preds_list.copy(), labels_list.copy()

            actual_episode_count += 1
            pbar.update(1)
            pbar.set_postfix({"avg_acc": f"{sum_acc/max(1, processed_eps):.1f}%"})

        pbar.close()

    # Rest of the function remains the same...
    # Average over episodes actually processed
    avg_acc = sum_acc / max(1, processed_eps)
    
    # Confusion matrix plotting and return statements...
    if best_labels:
        best_path = os.path.join(plot_dir, f"confmat_best_{mode_tag}.png")
        plot_confusion_from_preds(
            best_preds, best_labels,
            save_path=best_path,
            class_names=class_names,
            normalize=False,
            title=f"Best Episode · mode={mode_tag}"
        )
    else:
        print(f"[test:{mode_tag}] No valid best-episode data to plot.")

    if worst_labels:
        worst_path = os.path.join(plot_dir, f"confmat_worst_{mode_tag}.png")
        plot_confusion_from_preds(
            worst_preds, worst_labels,
            save_path=worst_path,
            class_names=class_names,
            normalize=False,
            title=f"Worst Episode · mode={mode_tag}"
        )
    else:
        print(f"[test:{mode_tag}] No valid worst-episode data to plot.")

    # Normalize sentinel values if nothing was evaluated
    if best_acc < 0:
        best_acc = 0.0
    if worst_acc > 100.0:
        worst_acc = 0.0

    print(f"[test:{mode_tag}] per-episode log -> {iter_csv}")
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

    # Class names for confusion matrix (unseen test classes)
    class_names = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

    # Determine ablation plan
    ablation_modes = cfg.get("ablation_modes")
    if not ablation_modes:
        # Single pass using boolean flags or full (default)
        ablation_modes = [cfg.get("ablation_mode", "full")]

    # Per-mode summary CSV
    log_dir = os.path.join(cfg["logs_dir"], f"{cfg['experiment_name']}")
    os.makedirs(log_dir, exist_ok=True)
    summary_csv = os.path.join(log_dir, "ablation_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerow(["mode", "best_acc", "avg_acc", "worst_acc"])

    summary = []
    for mode_tag in ablation_modes:
        # Fresh model per mode (ensures no contamination from fine-tuning in prior mode)
        model = DualSiameseNet(
            num_classes=cfg["n_way"],
            agg_method=cfg.get("agg_method", "mean"),
            device=device,
            relation_used_branches=cfg.get("train_relation_used_branches", ["S","D","C"])
        ).to(device)

        # Load best/latest weights
        load_checkpoint(model, cfg, optimizer=None, best=cfg.get("best", True))

        # Apply ablation
        _apply_ablation(model, mode_tag, cfg)

        # Evaluate this mode
        best_acc, avg_acc, worst_acc = _evaluate_one_mode(cfg, model, test_loader, class_names, mode_tag)
        summary.append((mode_tag, best_acc, avg_acc, worst_acc))

        # Append to summary CSV
        with open(summary_csv, "a", newline="") as f:
            csv.writer(f).writerow([mode_tag, f"{best_acc:.2f}", f"{avg_acc:.2f}", f"{worst_acc:.2f}"])

    # Print short summary
    print("\n=== Ablation Summary ===")
    for mode_tag, best_acc, avg_acc, worst_acc in summary:
        print(f"{mode_tag:>12} | Best: {best_acc:6.2f}% | Avg: {avg_acc:6.2f}% | Worst: {worst_acc:6.2f}%")
