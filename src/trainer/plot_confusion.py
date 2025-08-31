# src/trainer/plot_confusion.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _ensure_dir(path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    # avoid div-by-zero
    denom = cm.sum(axis=1, keepdims=False)
    denom[denom == 0] = 1
    return cm.diagonal() / denom


def _plot_confusion(cm: np.ndarray, class_names=None, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names if class_names is not None else None,
        yticklabels=class_names if class_names is not None else None
    )
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[*] Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def _acc_from_preds_labels(preds, labels) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.size == 0:
        return 0.0
    return 100.0 * float((preds == labels).sum()) / float(labels.size)


def compute_confusion_matrix_from_preds(preds, labels, save_path=None, class_names=None, title="Confusion Matrix"):
    """
    Compute & plot confusion matrix from numpy/int lists of predictions & labels.
    Also prints per-class accuracy.

    Args:
        preds: iterable of predicted class indices (length = N)
        labels: iterable of true class indices (length = N)
        save_path: where to save the image (optional)
        class_names: optional list of class names in label-index order
        title: title for the plot
    """
    preds_np = np.asarray(preds)
    labels_np = np.asarray(labels)
    cm = confusion_matrix(labels_np, preds_np)

    # Per-class accuracy
    per_class_acc = _per_class_accuracy(cm)
    print("\n[*] Per-Class Accuracy:")
    for idx, acc in enumerate(per_class_acc):
        cls_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"  {cls_name}: {acc*100:.2f}%")

    _plot_confusion(cm, class_names=class_names, title=title, save_path=save_path)
    return cm, per_class_acc


def compute_best_worst_confusions(episodes, class_names, out_dir, prefix=""):
    """
    From a list of episodes (preds/labels), find best and worst episodes
    by accuracy and plot/save their confusion matrices.

    Args:
        episodes: list of dicts OR tuples. Each element should be either:
                  - {"preds": list/np.array, "labels": list/np.array, "episode": int (optional)}
                  - (preds, labels)  # tuple
        class_names: list of class names
        out_dir: base directory for plots
        prefix: string added to filenames (e.g., ablation mode)

    Saves:
        <out_dir>/<prefix>_best_confusion.png
        <out_dir>/<prefix>_worst_confusion.png
    """
    if not episodes:
        print("[!] No episodes provided to compute_best_worst_confusions.")
        return None, None

    # Normalize to list of dicts
    norm_eps = []
    for i, item in enumerate(episodes):
        if isinstance(item, dict):
            preds = item["preds"]
            labels = item["labels"]
            ep_id = item.get("episode", i + 1)
        else:
            preds, labels = item
            ep_id = i + 1
        acc = _acc_from_preds_labels(preds, labels)
        norm_eps.append({"episode": ep_id, "preds": preds, "labels": labels, "acc": acc})

    # Identify best and worst by accuracy
    best_ep = max(norm_eps, key=lambda d: d["acc"])
    worst_ep = min(norm_eps, key=lambda d: d["acc"])

    print(f"[*] Best episode #{best_ep['episode']} | Acc: {best_ep['acc']:.2f}%")
    print(f"[*] Worst episode #{worst_ep['episode']} | Acc: {worst_ep['acc']:.2f}%")

    # Paths
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{prefix}_best_confusion.png" if prefix else "best_confusion.png")
    worst_path = os.path.join(out_dir, f"{prefix}_worst_confusion.png" if prefix else "worst_confusion.png")

    # Plot both
    compute_confusion_matrix_from_preds(
        best_ep["preds"], best_ep["labels"],
        save_path=best_path,
        class_names=class_names,
        title=f"Best Episode (#{best_ep['episode']}) · Acc {best_ep['acc']:.2f}%"
    )
    compute_confusion_matrix_from_preds(
        worst_ep["preds"], worst_ep["labels"],
        save_path=worst_path,
        class_names=class_names,
        title=f"Worst Episode (#{worst_ep['episode']}) · Acc {worst_ep['acc']:.2f}%"
    )

    return best_ep, worst_ep


# ----------------------------------------------------------------------
# Backward-compatible API (model + loader)
# ----------------------------------------------------------------------
def compute_confusion_matrix(model, loader, device, save_path=None, class_names=None):
    """
    Legacy compatibility: compute confusion from a simple classification loader.
    NOTE: This path expects your model to return (sim, dsim, logits) when called as model(img, img),
    which is how your older pipeline used it. For relation-based few-shot testing,
    prefer the *prediction-based* functions above.

    Args:
        model: trained model
        loader: DataLoader yielding (img, label)
        device: torch.device
        save_path: optional path for saving the plot
        class_names: optional names per class index
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)
            # legacy forward signature
            _, _, logits = model(img, img)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

    acc = _acc_from_preds_labels(all_preds, all_labels)
    print(f"[*] Overall Accuracy: {acc:.2f}%")

    compute_confusion_matrix_from_preds(
        all_preds, all_labels,
        save_path=save_path,
        class_names=class_names,
        title=f"Confusion Matrix · Acc {acc:.2f}%"
    )
