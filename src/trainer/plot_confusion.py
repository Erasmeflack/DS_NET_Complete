# src/trainer/plot_confusion.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def _ensure_class_names(class_names, num_classes, labels_present=None):
    if class_names and len(class_names) == num_classes:
        return class_names
    # Fallback to generic names if not provided / mismatched length
    if labels_present is None:
        labels_present = list(range(num_classes))
    return [str(c) for c in labels_present]

def plot_confusion_from_preds(
    preds,
    labels,
    save_path=None,
    class_names=None,
    normalize=False,
    title="Confusion Matrix"
):
    """
    Plot a confusion matrix from prediction/label lists.

    Args:
        preds (list[int]): predicted class indices
        labels (list[int]): ground-truth class indices
        save_path (str|None): if provided, save image here
        class_names (list[str]|None): optional names per class index
        normalize (bool): if True, row-normalize to show per-class accuracy
        title (str): plot title
    """
    
    # ---- NEW: guards for empties ----
    if preds is None or labels is None or len(preds) == 0 or len(labels) == 0:
        print("[confmat] Empty preds/labels; skipping confusion plot.")
        return

    # Derive label space from data to avoid mismatch
    unique_labels = sorted(set(labels) | set(preds))

    # ---- NEW: guard if still empty (paranoid) ----
    if len(unique_labels) == 0:
        print("[confmat] No unique labels present; skipping confusion plot.")
        return

    cm = confusion_matrix(labels, preds, labels=unique_labels)
    # Derive label space from data to avoid mismatch
    unique_labels = sorted(set(labels) | set(preds))
    cm = confusion_matrix(labels, preds, labels=unique_labels)

    if normalize:
        # Row-wise normalization; safe divide with zeros handled
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        fmt = ".2f"
        cbar_label = "Proportion"
    else:
        cm = cm.astype(np.int64)
        fmt = "d"
        cbar_label = "Count"

    # Per-class accuracy (diagonal / row sum); guard against zero rows
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True) if normalize else cm.sum(axis=1, keepdims=True).astype(np.float64)
        diag = np.diag(cm)
        # If not normalized, convert to proportions for reporting
        denom = row_sum.squeeze()
        denom[denom == 0] = 1.0
        per_class_acc = (diag / denom) if normalize else (diag / denom)

    # Resolve class names
    names = _ensure_class_names(class_names, num_classes=len(unique_labels), labels_present=unique_labels)

    print("\n[*] Per-Class Accuracy:")
    for idx, acc in enumerate(per_class_acc):
        cls_name = names[idx] if names and idx < len(names) else f"Class {idx}"
        print(f"  {cls_name}: {acc * 100:.2f}%")

    # Plot
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,              # match dtype
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": cbar_label}
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[*] Confusion matrix saved to {save_path}")
        plt.close()
    else:
        plt.show()
