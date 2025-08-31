# models/relation_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationModule(nn.Module):
    def __init__(self, input_channels=512, spatial_size=4, use_groupnorm=False, gn_groups=8):
        """
        Lightweight relation head: 2 conv blocks + MLP to a scalar score in [0,1].

        Args:
            input_channels (int): channels of the pairwise fusion map (default 512 after 1x1 reduce).
            spatial_size (int): spatial size of the input maps (default 4 for 64x64 pipeline).
            use_groupnorm (bool): use GroupNorm instead of BatchNorm for tiny episodic batches.
            gn_groups (int): number of groups for GroupNorm when enabled.
        """
        super().__init__()
        Norm2d = (lambda C: nn.GroupNorm(gn_groups, C)) if use_groupnorm else nn.BatchNorm2d

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) if spatial_size >= 4 else nn.Identity(),  # 4x4 -> 2x2
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) if spatial_size >= 3 else nn.Identity(),  # 2x2 -> 1x1
        )

        # infer flattened dim
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, spatial_size, spatial_size)
            flat_dim = self.conv_block2(self.conv_block1(dummy)).view(1, -1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
