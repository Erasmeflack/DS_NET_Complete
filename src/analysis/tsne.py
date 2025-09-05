# src/analysis/tsne.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

@torch.no_grad()
def _to_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:  # [C,H,W] -> [1,C,H,W]
        x = x.unsqueeze(0)
    if x.dim() == 5 and x.size(0) == 1:  # [1,B,C,H,W] -> [B,C,H,W]
        x = x.squeeze(0)
    return x

@torch.no_grad()
def _get_one_episode(loader, device):
    """
    Returns one episode tensors normalized to:
      support_imgs: [S,1,64,64], support_labels: [S]
      query_imgs:   [Q,1,64,64], query_labels:   [Q]
    """
    support_imgs, support_labels, query_imgs, query_labels = next(iter(loader))
    support_imgs  = _to_4d(support_imgs).to(device, non_blocking=True)
    query_imgs    = _to_4d(query_imgs).to(device, non_blocking=True)
    support_labels= support_labels.view(-1).to(device, non_blocking=True)
    query_labels  = query_labels.view(-1).to(device, non_blocking=True)
    return support_imgs, support_labels, query_imgs, query_labels

@torch.no_grad()
def _extract_branch_feats(model, images, branch_name: str):
    """
    Returns (N, D) features from a specific branch using forward_single.
    branch_name in {"similarity","dissimilarity","classification"}.
    """
    if branch_name == "similarity":
        # top_branch
        flats, _maps = model.top_branch.forward_single(images)
    elif branch_name == "dissimilarity":
        # bottom_branch
        flats, _maps = model.bottom_branch.forward_single(images)
    elif branch_name == "classification":
        # mid_branch
        flats, _maps = model.mid_branch.forward_single(images)
    else:
        raise ValueError(f"Unknown branch: {branch_name}")
    return flats.detach().cpu().numpy()  # (N, 4096)

def _run_tsne(feats_np: np.ndarray, perplexity: int = 15, seed: int = 1):
    """
    feats_np: (N, D) -> returns (N, 2)
    """
    n = feats_np.shape[0]
    # keep TSNE stable with small perplexity for small N
    perp = min(perplexity, max(5, (n - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate='auto',
                init='pca', random_state=seed, n_iter=1500, verbose=0)
    emb2d = tsne.fit_transform(feats_np)
    return emb2d

def _plot_tsne(emb2d: np.ndarray, labels_np: np.ndarray, class_names: list, title: str, save_path: str):
    """
    Save a 2D scatter. One point per sample, colored by class name, with legend dynamically placed.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        idx = labels_np == i
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], label=class_name, alpha=0.6, s=50)

    # Dynamic legend placement
    # Compute the bounding box of the t-SNE points
    x_min, x_max = np.min(emb2d[:, 0]), np.max(emb2d[:, 0])
    y_min, y_max = np.min(emb2d[:, 1]), np.max(emb2d[:, 1])
    
    # Define four possible legend positions (corners) in axes coordinates
    corners = [
        ('upper right', (0.95, 0.95)),
        ('upper left', (0.05, 0.95)),
        ('lower right', (0.95, 0.05)),
        ('lower left', (0.05, 0.05))
    ]
    
    # Calculate point density near each corner
    def count_points_near_corner(x, y, corner_x, corner_y, threshold=0.1):
        # Count points within a threshold distance from the corner (in normalized axes coords)
        distances = np.sqrt(((emb2d[:, 0] - x_min) / (x_max - x_min) - corner_x)**2 +
                            ((emb2d[:, 1] - y_min) / (y_max - y_min) - corner_y)**2)
        return np.sum(distances < threshold)
    
    # Find the corner with the fewest points nearby
    min_points = float('inf')
    best_corner = corners[0]
    for loc, (x, y) in corners:
        points = count_points_near_corner(x, y, x, y)
        if points < min_points:
            min_points = points
            best_corner = (loc, (x, y))
    
    # Place legend at the corner with the least density
    plt.legend(fontsize=10, loc=best_corner[0], bbox_to_anchor=best_corner[1])
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved t-SNE plot at {save_path}")

@torch.no_grad()
def tsne_one_val_episode(model, val_loader, cfg, tag: str = "start"):
    """
    Extracts features for support+query of ONE validation episode and saves TSNE plots
    for the three branches:
      - similarity (top_branch)
      - dissimilarity (bottom_branch)
      - classification (mid_branch)

    tag: e.g., "start" (before training), "end" (after/best).
    Outputs:
      ./plots/exp_<name>/tsne_<tag>_<branch>.png
    """
    device = cfg["device"]
    exp = cfg["experiment_name"]
    plot_dir = os.path.join(cfg.get("plot_dir","./plots"), f"{exp}")
    os.makedirs(plot_dir, exist_ok=True)

    # 1) grab one episode
    S_img, S_lab, Q_img, Q_lab = _get_one_episode(val_loader, device)
    images = torch.cat([S_img, Q_img], dim=0)               # (S+Q,1,64,64)
    labels = torch.cat([S_lab, Q_lab], dim=0).cpu().numpy() # (S+Q,)

    # 2) Get class names from the dataset
    val_dir = os.path.join(cfg["data_dir"], "train")
    from data.load_data import MSTARFSEpisodicDataset
    dataset = MSTARFSEpisodicDataset(val_dir, n_way=cfg["n_way"], k_shot=cfg["k_shot"], q_query=cfg.get("eval_q_query", cfg["q_query"]))
    class_names = dataset.classes  # List of class directory names

    # 3) branches
    for branch in ["similarity","dissimilarity","classification"]:
        feats = _extract_branch_feats(model, images, branch)    # (N,4096)
        emb2d = _run_tsne(feats, perplexity=15, seed=int(cfg.get("random_seed",1)))
        save_path = os.path.join(plot_dir, f"tsne_{tag}_{branch}.png")
        _plot_tsne(emb2d, labels, class_names, f't-SNE Visualization for SAR ATR ({branch} · {tag})', save_path)