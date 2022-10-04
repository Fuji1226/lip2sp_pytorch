"""
MixConv
Depthwise Convolutionでチャンネルを分割して異なるカーネルサイズで畳み込み
"""

import sys
from pathlib import Path

sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbam import SpatialAttention, ChannnelAttention
from net import NormLayer3D, NormalConv
from invres import InvResLayer3D
from utils import count_params


class MDConv(nn.Module):
    def __init__(self, inner_channels, n_groups):
        super().__init__()
        self.split_channels = inner_channels // n_groups
        layers = []
        for i in range(n_groups):
            kernel_size = int(2 * i + 3)
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv3d(self.split_channels, self.split_channels, kernel_size=(kernel_size, kernel_size, 1), padding=(padding, padding, 0), groups=self.split_channels)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x_split = torch.split(x, self.split_channels, dim=1)
        out = torch.cat([layer(input) for input, layer in zip(x_split, self.layers)], dim=1)
        return out


class InvResLayerMD(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, norm_type, up_scale=6, sq_r=16, kernel_size=None, pooling=True, c_attn=True, s_attn=True):
        super().__init__()
        self.hidden_channels = int(in_channels * up_scale)

        self.split_channels = self.hidden_channels // n_groups
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(self.hidden_channels, norm_type),
            nn.ReLU(),
        )
        self.depthwise_conv = nn.Sequential(
            MDConv(self.hidden_channels, n_groups),
            NormLayer3D(self.hidden_channels, norm_type),
        )
        if pooling:
            self.pool_layer = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        if c_attn:
            self.c_attention = ChannnelAttention(self.hidden_channels, sq_r)

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv3d(self.hidden_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )

        if s_attn:
            self.s_attention = SpatialAttention(kernel_size)

        if in_channels != out_channels:
            self.adjust_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                NormLayer3D(out_channels, norm_type),
            )

    def forward(self, x):
        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)
        
        if hasattr(self, "pool_layer"):
            out = self.pool_layer(out)
            x = self.pool_layer(x)

        if hasattr(self, "c_attention"):
            out = self.c_attention(out)

        out = self.pointwise_conv2(out)

        if hasattr(self, "s_attention"):
            out = self.s_attention(out)

        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        
        return F.relu(out + x)


class InvResNetMD(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, norm_type),
            nn.Dropout(dropout),

            InvResLayerMD(inner_channels, inner_channels + 16, 3, norm_type, kernel_size=7),
            nn.Dropout(dropout),

            InvResLayerMD(inner_channels + 16, inner_channels + 32, 3, norm_type, kernel_size=5),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels + 32, inner_channels + 48, norm_type, s_attn=False),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d(inner_channels + 48, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    net = InvResNetMD(1, 128, 16, 3, 0.1, "bn")
    x = torch.rand(1, 1, 48, 48, 150)
    out = net(x)
    print(out.shape)
    count_params(net, "net")
    
