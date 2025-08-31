# src/trainer/validate.py

import torch
from itertools import islice
from src.trainer.utils import relation_loss


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """Normalize image tensors to [B, C, H, W] by squeezing a leading episode dim if present."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() == 5 and x.size(0) == 1:
        x = x.squeeze(0)
    return x


def _split_batch_episode(support_imgs, support_labels, query_imgs, query_labels):
    """
    Normalizes shapes so downstream code can iterate over episodes uniformly.

    Supports:
      - batch_size == 1: tensors are [N*K, C, H, W] and [N*K], [N*Q, C, H, W], [N*Q]
      - batch_size  > 1: tensors are [B, N*K, C, H, W] and [B, N*K], [B, N*Q, C, H, W], [B, N*Q]
    Returns a list of per-episode tuples.
    """
    # Case 1: batch of 1 episode (no leading batch dim)
    if support_imgs.dim() == 4:
        return [(support_imgs, support_labels, query_imgs, query_labels)]

    # Case 2: batch of multiple episodes
    episodes = []
    B = support_imgs.size(0)
    for b in range(B):
        episodes.append((
            support_imgs[b],   # [N*K, C, H, W]
            support_labels[b], # [N*K]
            query_imgs[b],     # [N*Q, C, H, W]
            query_labels[b],   # [N*Q]
        ))
    return episodes


def validate(model, loader, cfg, epoch, val_log_path=None, verbose=True):
    """
    Episodic validation (no fine-tuning). Uses support-query relation scores to predict query labels.

    Args:
        model: Dual Siamese Network with forward(query, support) -> relation_scores, (optional logits)
        loader: episodic DataLoader yielding (support_imgs, support_labels, query_imgs, query_labels)
        cfg: dict with at least {"device", "n_way"}
        epoch: current epoch index (for logging)
        val_log_path: optional CSV path "epoch,acc,loss"
        verbose: print summary line
    """
    device = cfg["device"]

    # IMPORTANT: never fine-tune during training validation
    if cfg.get("fine_tune", False) and verbose:
        print("[validate] Note: fine_tune=True in cfg, but fine-tuning is disabled during validation.")

    # Cap validation workload per epoch
    max_eps = int(cfg.get("val_episodes", 200))

    model.eval()
    total_correct = 0
    total_queries = 0
    total_loss = 0.0
    num_episodes = 0

    with torch.no_grad():
        for i, (support_imgs, support_labels, query_imgs, query_labels) in islice(enumerate(loader), 0, max_eps):
            # Normalize shapes to per-episode tuples
            episodes = _split_batch_episode(_to_4d(support_imgs), support_labels, _to_4d(query_imgs), query_labels)

            for (S_img, S_lab, Q_img, Q_lab) in episodes:
                # Move to device and flatten labels
                S_img = S_img.to(device, non_blocking=True)
                Q_img = Q_img.to(device, non_blocking=True)
                S_lab = S_lab.view(-1).to(device, non_blocking=True)
                Q_lab = Q_lab.view(-1).to(device, non_blocking=True)

                # Forward relation evaluation: (Q, S, 1)
                relation_scores, _ = model(Q_img, S_img)

                # Predict class per query via model helper
                preds = model.predict_from_relations(relation_scores, S_lab)  # shape: [Q]

                # Accuracy
                total_correct += (preds == Q_lab).sum().item()
                total_queries += Q_lab.numel()

                # Optional monitoring loss (not used for backprop here)
                ep_loss = relation_loss(relation_scores, Q_lab, S_lab, num_classes=cfg["n_way"])
                total_loss += ep_loss.item()
                num_episodes += 1

    acc = 100.0 * total_correct / max(1, total_queries)
    avg_loss = total_loss / max(1, num_episodes)

    if verbose:
        print(f"[Validation] Epoch {epoch} | Episodes: {num_episodes} | Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    if val_log_path:
        with open(val_log_path, "a") as f:
            f.write(f"{epoch},{acc:.2f},{avg_loss:.4f}\n")

    return acc
