import sys
from pathlib import Path
import os
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F
from rnn_atten import RNNAttention


class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1, stride, stride), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        x = self.layers(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            NormalConv(in_channels, out_channels, stride),
            NormalConv(out_channels, out_channels, 1),
        )
        if stride > 1:
            self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if in_channels != out_channels:
            self.adjust_layer = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.layers(x)

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)

        return out + x


class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, dropout):
        super().__init__()
        self.conv3d = nn.ModuleList([
            NormalConv(in_channels, inner_channels, 2),
            nn.Dropout(dropout),

            ResBlock(inner_channels, inner_channels * 2, 2),            
            nn.Dropout(dropout),

            ResBlock(inner_channels * 2, inner_channels * 4, 2),
            nn.Dropout(dropout),
            
            ResBlock(inner_channels * 4, inner_channels * 8, 2),
            nn.Dropout(dropout),
        ])
        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        fmaps = []

        for layer in self.conv3d:
            x = layer(x)
            fmaps.append(x)

        x = torch.mean(x, dim=(3, 4))
        x = self.out_layer(x)   # (B, C, T)
        return x, fmaps


class AttentionResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, dropout, reduction_factor):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2),
            nn.Dropout(dropout),

            ResBlock(inner_channels, inner_channels * 2, 2),            
            nn.Dropout(dropout),

            RNNAttention(inner_channels * 2, dropout, reduction_factor),

            ResBlock(inner_channels * 2, inner_channels * 4, 2),
            nn.Dropout(dropout),

            RNNAttention(inner_channels * 4, dropout, reduction_factor),
            
            ResBlock(inner_channels * 4, inner_channels * 8, 2),
            nn.Dropout(dropout),

            RNNAttention(inner_channels * 8, dropout, reduction_factor),
        )
        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        out = self.conv3d(x)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)   # (B, C, T)
        return out