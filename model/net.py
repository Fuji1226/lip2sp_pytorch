import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
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
        return self.layers(x)


class MultiDilatedConv(nn.Module):
    def __init__(self, inner_channels, stride, n_groups):
        super().__init__()
        assert inner_channels % n_groups == 0
        self.split_channels = inner_channels // n_groups

        layers = []
        for g in range(n_groups):
            dilation = g + 1
            padding = dilation
            layers.append(
                nn.Conv3d(self.split_channels, self.split_channels, kernel_size=3, stride=(1, stride, stride), dilation=(dilation, 1, 1), padding=(padding, 1, 1), bias=False),
            )
        self.layers = nn.ModuleList(layers)

        self.bn = nn.BatchNorm3d(inner_channels)

    def forward(self, x):
        x_split = torch.split(x, self.split_channels, dim=1)

        out = []
        for each_x, layer in zip(x_split, self.layers):
            out.append(layer(each_x))
        
        out = torch.cat(out, dim=1)
        out = F.relu(self.bn(out))
        return out


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


class MultiDilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_groups):
        super().__init__()
        self.layers = nn.Sequential(
            NormalConv(in_channels, out_channels, stride),
            MultiDilatedConv(out_channels, 1, n_groups),
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
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2),
            nn.Dropout(dropout),

            ResBlock(inner_channels, inner_channels * 2, 2),            
            nn.Dropout(dropout),

            ResBlock(inner_channels * 2, inner_channels * 4, 2),
            nn.Dropout(dropout),
            
            ResBlock(inner_channels * 4, inner_channels * 8, 2),
            nn.Dropout(dropout),
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


class MultiDilatedResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, dropout, n_groups):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2),
            nn.Dropout(dropout),

            MultiDilatedResBlock(inner_channels, inner_channels * 2, 2, n_groups),            
            nn.Dropout(dropout),

            MultiDilatedResBlock(inner_channels * 2, inner_channels * 4, 2, n_groups),
            nn.Dropout(dropout),
            
            MultiDilatedResBlock(inner_channels * 4, inner_channels * 8, 2, n_groups),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        out = self.conv3d(x)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)   # (B, C, T)
        return out


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