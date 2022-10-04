import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F

from cbam import SpatialAttention, ChannnelAttention
from net import NormLayer3D


class DSLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, pooling=True):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), groups=in_channels),
            NormLayer3D(in_channels, norm_type),
        )
    
        if pooling:
            self.pool_layer = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.depthwise_conv(x)

        if hasattr(self, "pool_layer"):
            out = self.pool_layer(out)

        out = self.pointwise_conv(out)
        return out


class DSBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, norm_type, pooling=True):
        super().__init__()
        self.layer = nn.Sequential(
            DSLayer3D(in_channels, hidden_channels, norm_type, pooling=pooling),
            DSLayer3D(hidden_channels, out_channels, norm_type, pooling=False),
        )

        if pooling:
            self.pool_layer = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        if in_channels != out_channels:
            self.adjust_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                NormLayer3D(out_channels, norm_type),
                nn.ReLU(),
            )

    def forward(self, x):
        out = self.layer(x)

        if hasattr(self, "pool_layer"):
            x = self.pool_layer(x)
        
        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        return out + x


class DSBlockCbam(DSBlock):
    def __init__(self, in_channels, hidden_channels, out_channels, norm_type, sq_r, kernel_size, pooling=True):
        super().__init__(in_channels, hidden_channels, out_channels, norm_type, pooling)
        self.c_attn = ChannnelAttention(out_channels, sq_r)
        self.s_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.layer(x)
        out = self.c_attn(out)
        out = self.s_attn(out)

        if hasattr(self, "pool_layer"):
            x = self.pool_layer(x)
        
        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        return out + x


class DSResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            DSBlock(in_channels, inner_channels, inner_channels, norm_type),
            nn.Dropout(dropout),

            DSBlock(inner_channels, inner_channels * 2, inner_channels * 2, norm_type),
            nn.Dropout(dropout),

            DSBlock(inner_channels * 2, inner_channels * 4, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            
            DSBlock(inner_channels * 4, inner_channels * 8, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class DSResNet3DCbam(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            DSBlockCbam(in_channels, inner_channels, inner_channels, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels, inner_channels * 2, inner_channels * 2, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels * 2, inner_channels * 4, inner_channels * 4, norm_type, sq_r=16, kernel_size=5),
            nn.Dropout(dropout),
            
            DSBlock(inner_channels * 4, inner_channels * 8, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class DSResNet3DCbamSmall(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            DSBlockCbam(in_channels, inner_channels, inner_channels, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels, inner_channels * 2, inner_channels * 2, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels * 2, inner_channels * 4, inner_channels * 4, norm_type, sq_r=16, kernel_size=5),
            nn.Dropout(dropout),
            
            DSBlock(inner_channels * 4, inner_channels * 8, out_channels, norm_type),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        return out
