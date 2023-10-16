import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=(stride, stride, 1)),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if stride > 1:
            self.pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0))

        if in_channels != out_channels:
            self.adjust_layer = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        out = self.layer(x)

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)

        return out + x


class ResNet18(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(7, 7, 5), stride=(2, 2, 1), padding=(3, 3, 2)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.convs = nn.ModuleList([
            ResBlock(hidden_channels, hidden_channels, 1, dropout),
            ResBlock(hidden_channels, hidden_channels, 1, dropout),

            ResBlock(hidden_channels, hidden_channels * 2, 2, dropout),
            ResBlock(hidden_channels * 2, hidden_channels * 2, 1, dropout),

            ResBlock(hidden_channels * 2, hidden_channels * 4, 2, dropout),
            ResBlock(hidden_channels * 4, hidden_channels * 4, 1, dropout),

            ResBlock(hidden_channels * 4, hidden_channels * 8, 2, dropout),
            ResBlock(hidden_channels * 8, hidden_channels * 8, 1, dropout),
        ])

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        fmaps = []

        x = self.conv1(x)
        fmaps.append(x)

        x = self.pool1(x)
        fmaps.append(x)

        for layer in self.convs:
            x = layer(x)

        x = torch.mean(x, dim=(2, 3))
        return x, fmaps
    