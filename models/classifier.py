# models/classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MidLevelClassifier(nn.Module):
    def __init__(self, input_channels=768, hidden_channels=1024, spatial_size=4, num_classes=5, p_dropout=0.5):
        """
        Mid-level classifier for DS-Net.
        Expects fused query feature map with shape (B, 768, 4, 4) = concat([S-Net, D-Net, C-Net] maps).

        Args:
            input_channels (int): Number of input channels (default: 768 for S/D/M concat).
            hidden_channels (int): Channels of the conv layer.
            spatial_size (int): Spatial size of input maps (default: 4 for 64x64 input).
            num_classes (int): Number of classes.
            p_dropout (float): Dropout probability.
        """
        super(MidLevelClassifier, self).__init__()

        # Keep spatial size (padding=1) so flat_dim is predictable from spatial_size
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)

        # Infer flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, spatial_size, spatial_size)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Linear(flat_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 768, 4, 4)
        """
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
