import sys
from pathlib import Path
import os
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F
from attention_pooling import Encoder


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


class ResNet3DRemake(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, dropout, is_large):
        super().__init__()
        self.first_conv = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            ResBlock(inner_channels, inner_channels * 2, 2),            
            nn.Dropout(dropout),

            ResBlock(inner_channels * 2, inner_channels * 4, 2),
            nn.Dropout(dropout),
            
            ResBlock(inner_channels * 4, inner_channels * 8, 2),
            nn.Dropout(dropout),
        ])
        if is_large:
            self.final_conv = nn.Sequential(
                ResBlock(inner_channels * 8, inner_channels * 8, 2),
                nn.Dropout(dropout),
            )
            self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)
        else:
            self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        fmaps = []
        x = self.first_conv(x)
        fmaps.append(x)

        for layer in self.convs:
            x = layer(x)

        if hasattr(self, "final_conv"):
            x = self.final_conv(x)
            
        x = torch.mean(x, dim=(3, 4))
        x = self.out_layer(x)
        return x, fmaps


class ResNet3DVTP(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, dropout):
        super().__init__()
        self.conv3d = nn.ModuleList([
            NormalConv(in_channels, inner_channels, 2),
            nn.Dropout(dropout),

            ResBlock(inner_channels, inner_channels * 2, 2),
            nn.Dropout(dropout),

            ResBlock(inner_channels * 2, inner_channels * 4, 2),
            nn.Dropout(dropout),

            NormalConv(inner_channels * 4, inner_channels * 8, 1),
            nn.Dropout(dropout),
        ])
        self.attetion = nn.ModuleList([
            Encoder(2, 4, inner_channels * 8),
        ])
        self.q_att = nn.Parameter(torch.randn(inner_channels * 8, 1), requires_grad=True)
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

        x = x.permute(0, 2, 3, 4, 1)    # (B, T, H, W, C)
        for layer in self.attetion:
            x = layer(x)

        B, T, H, W, C = x.shape
        x = x.reshape(B, T, H * W, C)   # (B, T, H * W, C)
        a_t = torch.matmul(x, self.q_att)   # (B, T, H * W, 1)
        a_t = F.softmax(a_t, dim=-2)
        x = torch.matmul(x.permute(0, 1, 3, 2), a_t).squeeze(-1)    # (B, T, C)
        x /= (H * W)
        x = self.out_layer(x.permute(0, 2, 1))  # (B, C, T)
        return x, fmaps


if __name__ == "__main__":
    net = ResNet3DVTP(1, 256, 32, 0.1)
    x = torch.rand(1, 1, 48, 48, 150)
    out = net(x)    
