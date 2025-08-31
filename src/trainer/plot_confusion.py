import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def compute_confusion_matrix(model, loader, device, save_path=None, class_names=None):
    """
    Computes confusion matrix with class names and per-class accuracy.

    Args:
        model: Trained DualSiameseNet
        loader: DataLoader for query set
        device: torch.device
        save_path: Path to save confusion matrix image (optional)
        class_names: List of class names corresponding to labels (folder names)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)

            _, _, logits = model(img, img)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n[*] Per-Class Accuracy:")
    for idx, acc in enumerate(per_class_acc):
        cls_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"  {cls_name}: {acc*100:.2f}%")

    # Confusion matrix plot with class names
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"\n[*] Confusion matrix saved to {save_path}\n")
    else:
        plt.show()

    plt.close()
