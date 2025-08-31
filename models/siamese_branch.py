# models/siamese_branch.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseBranch(nn.Module):
    def __init__(self, use_groupnorm: bool = False, gn_groups: int = 8, dropout_p: float = 0.3):
        super(SiameseBranch, self).__init__()

        Norm2d = (lambda C: nn.GroupNorm(gn_groups, C)) if use_groupnorm else nn.BatchNorm2d

        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv1 = nn.Conv2d(1,   32, 5, padding=2)
        self.bn1   = Norm2d(32)
        self.conv2 = nn.Conv2d(32,  64, 5, padding=2)
        self.bn2   = Norm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = Norm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4   = Norm2d(256)

        # compute flat dim (expected 256*4*4 = 4096 for 64x64 input)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            fm = self.encode(dummy)
            flat_dim = fm.view(1, -1).shape[1]

        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(flat_dim, 4096)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1.0 / math.sqrt(max(1, fan_in))
                    nn.init.uniform_(m.bias, -bound, bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))  # 64->32
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))  # 32->16
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))  # 16->8
        x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2)))  # 8->4
        return x  # (B, 256, 4, 4)

    def project(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (B, 256, 4, 4) -> (B, 4096)
        return self.fc1(self.dropout(fmap.view(fmap.size(0), -1)))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Standard Siamese forward. Returns:
          f1_flat, f2_flat, f2_map
        """
        f1_map = self.encode(x1)
        # micro-opt: avoid recompute if tensors share storage (common in self-pairs)
        if x2.data_ptr() == x1.data_ptr():
            f2_map = f1_map
        else:
            f2_map = self.encode(x2)

        f1_flat = self.project(f1_map)
        f2_flat = self.project(f2_map)
        return f1_flat, f2_flat, f2_map

    def forward_single(self, x: torch.Tensor):
        fmap = self.encode(x)
        f_flat = self.project(fmap)
        return f_flat, fmap

    def pair_encode(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Helper used by episodic loops to minimize recompute when x1==x2.
        Returns:
          (f1_flat, f1_map), (f2_flat, f2_map)
        """
        f1_map = self.encode(x1)
        if x2.data_ptr() == x1.data_ptr():
            f2_map = f1_map
        else:
            f2_map = self.encode(x2)
        return (self.project(f1_map), f1_map), (self.project(f2_map), f2_map)
