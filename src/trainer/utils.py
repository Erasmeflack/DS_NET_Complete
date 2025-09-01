# src/trainer/utils.py
import torch
import torch.nn.functional as F

def contrastive_loss(out1, out2, label, margin=1.0):
    """Contrastive loss for pairwise embeddings."""
    distance = F.pairwise_distance(out1, out2, p=2)
    loss = 0.5 * (label * torch.pow(distance, 2) +
                  (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss.mean()


def similarity_loss(f1, f2):
    """Similarity Loss: Encourage embeddings to be close for similar pairs."""
    distance = F.pairwise_distance(f1, f2, p=2)
    loss = torch.log(1 - torch.sigmoid(distance) + 1e-8)
    return -loss.mean()


def dissimilarity_loss(f1, f2, margin=3.5):
    """Dissimilarity Loss: Penalize distances above margin."""
    distance = torch.norm(f1 - f2, p=2, dim=1)
    loss = torch.relu(distance - margin) ** 2
    return torch.mean(loss)


def relation_loss(relation_scores, query_labels, support_labels, num_classes, agg_method="mean"):
    """
    Relation-based loss for unified training.
    Computes cross-entropy on aggregated relation scores per class.

    Inputs:
      relation_scores: (Q, S, 1) or (Q, S)
      query_labels:    (Q,) or (1, Q)
      support_labels:  (S,) or (1, S)
    """
    # Normalize shapes
    if relation_scores.dim() == 3 and relation_scores.size(-1) == 1:
        relation_scores = relation_scores.squeeze(-1)  # (Q, S)
    support_labels = support_labels.view(-1)  # (S,)
    query_labels = query_labels.view(-1)  # (Q,)

    Q, S = relation_scores.shape
    device = relation_scores.device
    aggregated = torch.zeros(Q, num_classes, device=device)

    # Aggregate per class over support
    for c in range(num_classes):
        mask = (support_labels == c)  # (S,)
        if mask.any():
            per_class = relation_scores[:, mask]  # (Q, S_c)
            if agg_method == "max":
                agg_vals, _ = per_class.max(dim=1)  # (Q,)
            else:  # mean
                agg_vals = per_class.mean(dim=1)  # (Q,)
            aggregated[:, c] = agg_vals

    return F.cross_entropy(aggregated, query_labels)


def log_losses(log_paths, epoch, sim_losses, dsim_losses, cls_losses, rel_losses, total_losses=None):
    """Unify to a single CSV with epoch means."""
    def mean(xs):
        return float(sum(xs) / max(1, len(xs)))

    sim = mean(sim_losses)
    dis = mean(dsim_losses)
    cls = mean(cls_losses)
    rel = mean(rel_losses)
    uni = mean(total_losses) if total_losses is not None else (rel + sim + dis + cls)

    # Single file: loss.csv
    with open(log_paths["loss_log"], "a") as f:
        f.write(f"{epoch},{sim:.6f},{dis:.6f},{cls:.6f},{rel:.6f},{uni:.6f}\n")

    print(f"Epoch {epoch} | Sim Loss: {sim:.4f}, Dissim Loss: {dis:.4f}, Cls Loss: {cls:.4f}, Relation Loss: {rel:.4f}, Total(uni): {uni:.4f}")
