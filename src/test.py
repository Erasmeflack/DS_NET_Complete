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


def _fine_tune_on_support(model, support_imgs, support_labels, query_imgs, query_labels, cfg, device):
    """Fine-tune relation/C-Net on the episode's support set (optionally queries too)."""
    if not cfg.get("fine_tune", False):
        return

    # Freeze all first, then unfreeze selected
    for p in model.parameters():
        p.requires_grad = False

    params = []
    if cfg.get("fine_tune_classifier", True):
        for p in model.classifier.parameters():
            p.requires_grad = True
        params += list(model.classifier.parameters())
    if cfg.get("fine_tune_relation", False):
        for p in model.relation_module.parameters():
            p.requires_grad = True
        params += list(model.relation_module.parameters())

    if not params:
        return

    opt_name = cfg.get("fine_tune_opt", "adam").lower()
    lr = cfg.get("fine_tune_lr", 5e-5)
    optimizer = getattr(torch.optim, opt_name.capitalize())(params, lr=lr)

    model.train()
    epochs = cfg.get("fine_tune_epochs", 5)
    for _ in range(epochs):
        optimizer.zero_grad()
        rel_scores, _ = model(query_imgs, support_imgs)
        loss = relation_loss(rel_scores, query_labels, support_labels, cfg["n_way"])
        loss.backward()
        optimizer.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def test(cfg):
    device = cfg.get("device", "cuda" if cfg.get("use_gpu", True) and torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    # Model
    model = DualSiameseNet(num_classes=cfg["n_way"], agg_method=cfg.get("agg_method", "mean"), device=device).to(device)

    # Load best/latest
    load_checkpoint(model, cfg, optimizer=None, best=cfg.get("best", True))

    # Episodic test loader
    test_data_dir = os.path.join(cfg["data_dir"], "test")
    test_loader = get_mstar_dataloader(
        root_dir=test_data_dir,
        n_way=cfg["n_way"],
        k_shot=cfg["k_shot"],
        q_query=cfg["q_query"],
        batch_size=cfg.get("eval_batch_size", 1),
        augment=False,
    )

    # Class names for confusion matrix
    class_names = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

    num_episodes = cfg.get("num_episodes", 1000)
    best_acc = 0.0
    total_acc = 0.0
    best_preds, best_labels = [], []

    # CSV logging
    log_dir = os.path.join(cfg["logs_dir"], f"exp_{cfg['experiment_name']}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "test_accuracy.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "accuracy", "timestamp"]) 

    print(f"[*] Evaluating over {num_episodes} few-shot episodes...")

    # Iterate episodes
    ep_iter = _episodic_iter(test_loader)
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        try:
            support_imgs, support_labels, query_imgs, query_labels = next(ep_iter)
        except StopIteration:
            # Reinitialize iterator if exhausted
            ep_iter = _episodic_iter(test_loader)
            support_imgs, support_labels, query_imgs, query_labels = next(ep_iter)

        support_imgs  = support_imgs.to(device)
        support_labels= support_labels.to(device)
        query_imgs    = query_imgs.to(device)
        query_labels  = query_labels.to(device)

        # Optional per-episode fine-tune
        if cfg.get("fine_tune", False):
            _fine_tune_on_support(model, support_imgs, support_labels, query_imgs, query_labels, cfg, device)

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

        # Track best episode
        if ep_acc > best_acc:
            best_acc = ep_acc
            best_preds = preds_np
            best_labels = labels_np

    avg_acc = total_acc / max(1, num_episodes)

    print("\n[*] Evaluation complete")
    print(f"[*] Best Episode Accuracy: {best_acc:.2f}%")
    print(f"[*] Average Accuracy over {num_episodes} episodes: {avg_acc:.2f}%")

    # Confusion matrix for the best-performing episode
    print("\n[*] Plotting confusion matrix for best episode...")
    compute_confusion_matrix(best_preds, best_labels, save_path=f"./plots/{cfg['experiment_name']}_best_confusion.png", class_names=class_names)
