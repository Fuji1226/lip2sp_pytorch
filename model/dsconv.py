import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F

from cbam import SpatialAttention, ChannnelAttention
from net import NormLayer3D, NormalConv
from utils import count_params


class DSLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), groups=in_channels),
            NormLayer3D(in_channels, norm_type),
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            NormLayer3D(out_channels, norm_type),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class DSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type):
        super().__init__()
        self.layer = nn.Sequential(
            DSLayer3D(in_channels, out_channels, stride, norm_type),
            DSLayer3D(out_channels, out_channels, 1, norm_type),
        )

        if stride > 1:
            self.pool_layer = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if in_channels != out_channels:
            self.adjust_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                NormLayer3D(out_channels, norm_type),
            )

    def forward(self, x):
        out = self.layer(x)

        if hasattr(self, "pool_layer"):
            x = self.pool_layer(x)
        
        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        return out + x


class DSResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type, sq_r, attn):
        super().__init__()
        self.conv1 = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            DSBlock(inner_channels, inner_channels * 2, 2, norm_type),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            DSBlock(inner_channels * 2, inner_channels * 4, 2, norm_type),
            nn.Dropout(dropout),
        )
        self.conv4 = nn.Sequential(
            DSBlock(inner_channels * 4, inner_channels * 8, 2, norm_type),
            nn.Dropout(dropout),
        )
        if attn:
            self.attn1 = nn.Sequential(
                ChannnelAttention(inner_channels * 2, sq_r),
                SpatialAttention(kernel_size=7),
            )
            self.attn2 = nn.Sequential(
                ChannnelAttention(inner_channels * 4, sq_r),
                SpatialAttention(kernel_size=5),
            )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)

        out = self.conv1(x)
        out = self.conv2(out)
        if hasattr(self, "attn1"):
            out = self.attn1(out)
        
        out = self.conv3(out)
        if hasattr(self, "attn2"):
            out = self.attn2(out)

        out = self.conv4(out)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    net = DSResNet3D(
        in_channels=5,
        out_channels=128,
        inner_channels=32,
        layers=3,
        dropout=0.1,
        norm_type="bn",
        sq_r=16,
        attn=True,
    )
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    print(out.shape)
    count_params(net, "net")