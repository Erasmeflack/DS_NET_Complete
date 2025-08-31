# src/trainer/train_loop.py

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from itertools import islice

from .utils import (
    similarity_loss as aux_similarity_loss,   # (f1, f2) style if you want it later
    dissimilarity_loss as aux_dissimilarity_loss,
    relation_loss as relation_ce_loss,
    log_losses,
)

# --- (optional local helper) ---
def _to_4d(x):
    # [C,H,W] -> [1,C,H,W], [1,B,C,H,W] -> [B,C,H,W]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() == 5 and x.size(0) == 1:
        x = x.squeeze(0)
    return x

def train_one_epoch(model, loader, optimizer, cfg, epoch, log_paths):
    """
    Joint multi-task training for one epoch (episodic).
    - Primary objective: relation-based classification (metric inference).
    - Auxiliary regularizers: similarity (S-Net), dissimilarity (D-Net),
      and A2 prototype-style episodic classification with ONLY C-Net features.

    Expected dataloader output per batch (one episode):
        support_images: [N*K, C, H, W]
        support_labels: [N*K]
        query_images:   [N*q, C, H, W]
        query_labels:   [N*q]
    """
    device   = cfg["device"]
    n_way    = int(cfg.get("n_way", 5))
    k_shot   = int(cfg.get("k_shot", 1))
    q_query  = int(cfg.get("q_query", 15))

    # Loss weights (regularizers vs. main task)
    lam_sim  = float(cfg.get("lambda_sim",    0.2))
    lam_dsim = float(cfg.get("lambda_dissim", 0.2))
    lam_cls  = float(cfg.get("lambda_cls",    0.2))
    lam_rel  = float(cfg.get("lambda_rel",    1.0))

    # Prototype temperature (scale for cosine logits)
    proto_temp = float(cfg.get("proto_temp", 10.0))

    model.train()

    # Running logs
    sim_losses, dsim_losses, cls_losses, rel_losses, total_losses = [], [], [], [], []

    pbar = tqdm(
        islice(enumerate(loader), 0, cfg.get("episodes_per_epoch", 1000)),
        total=cfg.get("episodes_per_epoch", 1000),
        desc=f"Epoch {epoch}", mininterval=1.0
    )

    bce = nn.BCELoss()

    for _, batch in pbar:
        support_imgs, support_labels, query_imgs, query_labels = batch

        # normalize shapes to episode tensors
        support_imgs   = _to_4d(support_imgs).to(device)     # [S, C, H, W]
        query_imgs     = _to_4d(query_imgs).to(device)       # [Q, C, H, W]
        support_labels = support_labels.view(-1).to(device)  # [S]
        query_labels   = query_labels.view(-1).to(device)    # [Q]

        optimizer.zero_grad()

        # -------------------------------
        # 1) RELATION LOSS (primary task)
        # -------------------------------
        relation_scores, _ = model(query_imgs, support_imgs)  # (Q, S, 1)
        rel_loss = relation_ce_loss(
            relation_scores=relation_scores,
            query_labels=query_labels,
            support_labels=support_labels,
            num_classes=n_way
        )

        # ---------------------------------------------------
        # 2) SIMILARITY LOSS (aux) using matched (q, s) pairs
        # ---------------------------------------------------
        sim_loss_accum = 0.0
        pos_pairs = 0

        # Build class -> indices map for supports
        class_to_sidx = {}
        for c in range(n_way):
            mask = (support_labels == c).nonzero(as_tuple=True)[0]
            if mask.numel() > 0:
                class_to_sidx[c] = mask

        # sample at most one positive support per query
        matched_support_idx = []
        for q_idx in range(query_labels.size(0)):
            c = int(query_labels[q_idx].item())
            if c in class_to_sidx:
                sidx = class_to_sidx[c][torch.randint(0, class_to_sidx[c].numel(), (1,)).item()]
                matched_support_idx.append((q_idx, sidx))

        for (q_idx, s_idx) in matched_support_idx:
            q_img = query_imgs[q_idx].unsqueeze(0)   # [1, C, H, W]
            s_img = support_imgs[s_idx].unsqueeze(0) # [1, C, H, W]
            sim_score, _, _ = model.forward_pairwise(s_img, q_img)  # (B,) scalar prob in [0,1]
            # maximize similarity for matched
            sim_loss_accum += bce(sim_score.view(-1), torch.ones_like(sim_score.view(-1)))
            pos_pairs += 1

        sim_loss = sim_loss_accum / max(pos_pairs, 1)

        # ------------------------------------------------------
        # 3) DISSIMILARITY LOSS (aux) using mismatched (q, s) pairs
        # ------------------------------------------------------
        dsim_loss_accum = 0.0
        neg_pairs = 0
        dsim_margin = float(cfg.get("dsim_margin", 1.0))  # margin in [0,2] after L2-norm

        for q_idx in range(query_labels.size(0)):
            cq = int(query_labels[q_idx].item())
            neg_indices = (support_labels != cq).nonzero(as_tuple=True)[0]
            if neg_indices.numel() == 0:
                continue
            s_idx = neg_indices[torch.randint(0, neg_indices.numel(), (1,)).item()]

            q_img = query_imgs[q_idx].unsqueeze(0)
            s_img = support_imgs[s_idx].unsqueeze(0)

            # Use bottom branch raw features (before any extra FC head)
            f_s_flat, f_q_flat, _ = model.bottom_branch(s_img, q_img)  # (1,4096) each

            # L2-normalize so distances are bounded in [0,2]
            f_s = F.normalize(f_s_flat, p=2, dim=1, eps=1e-12)
            f_q = F.normalize(f_q_flat, p=2, dim=1, eps=1e-12)

            dist = torch.norm(f_s - f_q, p=2, dim=1)  # (1,)
            # push-apart margin
            dsim_loss_accum += F.relu(dsim_margin - dist).pow(2).mean()
            neg_pairs += 1

        dsim_loss = dsim_loss_accum / max(neg_pairs, 1)

        # -----------------------------------------------------------------
        # 4) A2 PROTOTYPE-STYLE EPISODIC CLASSIFICATION (ONLY C-Net features)
        # -----------------------------------------------------------------
        # Extract flat embeddings from C-Net for support and query
        fS_flat, _ = model.mid_branch.forward_single(support_imgs)  # (S, 4096)
        fQ_flat, _ = model.mid_branch.forward_single(query_imgs)    # (Q, 4096)

        # Normalize for cosine similarity
        fS = F.normalize(fS_flat, p=2, dim=1, eps=1e-12)            # (S, D)
        fQ = F.normalize(fQ_flat, p=2, dim=1, eps=1e-12)            # (Q, D)

        # Build class prototypes (mean of supports per class)
        protos_list = []
        D = fS.size(1)
        for c in range(n_way):
            mask = (support_labels == c)
            if mask.any():
                protos_list.append(fS[mask].mean(0, keepdim=True))   # (1, D)
            else:
                # safety (should not happen with proper episodic sampling)
                protos_list.append(torch.zeros(1, D, device=device))
        protos = torch.cat(protos_list, dim=0)                       # (N, D)
        protos = F.normalize(protos, p=2, dim=1, eps=1e-12)

        # Cosine logits scaled by temperature
        logits = proto_temp * (fQ @ protos.T)                        # (Q, N)

        # Cross-entropy against query labels (support-aware)
        cls_loss = F.cross_entropy(logits, query_labels)

        # -----------------------
        # TOTAL LOSS & OPTIM STEP
        # -----------------------
        total_loss = lam_rel * rel_loss + lam_sim * sim_loss + lam_dsim * dsim_loss + lam_cls * cls_loss
        total_loss.backward()
        optimizer.step()

        # Accumulate logs
        sim_losses.append(sim_loss.detach().item())
        dsim_losses.append(dsim_loss.detach().item())
        cls_losses.append(cls_loss.detach().item())
        rel_losses.append(rel_loss.detach().item())
        total_losses.append(total_loss.detach().item())

        pbar.set_postfix({
            "Rel":  f"{sum(rel_losses)/len(rel_losses):.4f}",
            "Sim":  f"{sum(sim_losses)/len(sim_losses):.4f}",
            "Dsim": f"{sum(dsim_losses)/len(dsim_losses):.4f}",
            "Cls":  f"{sum(cls_losses)/len(cls_losses):.4f}",
            "Tot":  f"{sum(total_losses)/len(total_losses):.4f}",
        })

    # Persist logs (relation logged as 'unified'; total for completeness)
    log_losses(
        log_paths,
        epoch,
        sim_losses,
        dsim_losses,
        cls_losses,
        rel_losses,
        total_losses
    )
