# data/load_data.py

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict

# -------------------------------------------------------
#  Image loader with transforms
# -------------------------------------------------------
def default_loader(path):
    return Image.open(path).convert('L')

# -------------------------------------------------------
#  Episodic Few-Shot Dataset for MSTAR
# -------------------------------------------------------
class MSTARFSEpisodicDataset(Dataset):
    """
    Episodic dataset for Few-Shot Learning on MSTAR.
    Each __getitem__ returns a support set and a query set for one episode.
    """
    def __init__(self, root_dir, n_way=5, k_shot=1, q_query=15, augment=False, loader=default_loader,
                 num_workers=4, pin_memory=False, persistent_workers=False):
        self.root_dir = root_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.loader = loader

        # --- Transforms ---
        # Train-time: light & fast. Avoid GaussianBlur for 64x64 and keep content.
        # Val/Test: deterministic, no augmentation.
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.11975140869617462], std=[0.1193585991859436]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.11975140869617462], std=[0.1193585991859436]),
            ])

        # Build class-to-images mapping
        self.class_to_images = defaultdict(list)
        self.classes = sorted(os.listdir(root_dir))
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.class_to_images[cls].append(os.path.join(class_dir, fname))

    def __len__(self):
        # Treat one __getitem__ as one episode. Return a large virtual length.
        return 100000

    def __getitem__(self, idx):
        # 1) Sample N classes
        selected_classes = random.sample(self.classes, self.n_way)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for label, cls in enumerate(selected_classes):
            image_pool = self.class_to_images[cls]
            # Ensure enough images exist
            assert len(image_pool) >= self.k_shot + self.q_query, (
                f"Class {cls} has only {len(image_pool)} images; needs >= {self.k_shot + self.q_query}")

            image_paths = random.sample(image_pool, self.k_shot + self.q_query)

            # Split into support and query
            support_paths = image_paths[:self.k_shot]
            query_paths = image_paths[self.k_shot:]

            for path in support_paths:
                img = self.transform(self.loader(path))
                support_images.append(img)
                support_labels.append(label)

            for path in query_paths:
                img = self.transform(self.loader(path))
                query_images.append(img)
                query_labels.append(label)

        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        return support_images, support_labels, query_images, query_labels

# -------------------------------------------------------
#  DataLoader helpers
# -------------------------------------------------------
def get_mstar_dataloader(root_dir, n_way=5, k_shot=1, q_query=15, batch_size=1, augment=False,
                         num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """
    Returns episodic dataloader for MSTAR dataset.
    Each batch = 1 episode (support + query sets).
    """
    dataset = MSTARFSEpisodicDataset(
        root_dir=root_dir,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
