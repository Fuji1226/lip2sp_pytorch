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


class InvResLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type, up_scale, sq_r, kernel_size=None, c_attn=True, s_attn=True):
        super().__init__()
        self.hidden_channels = int(in_channels * up_scale)
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            NormLayer3D(self.hidden_channels, norm_type),
            nn.ReLU(),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), groups=self.hidden_channels),
            NormLayer3D(self.hidden_channels, norm_type),
        )

        if c_attn:
            self.c_attention = ChannnelAttention(self.hidden_channels, sq_r)

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv3d(self.hidden_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            NormLayer3D(out_channels, norm_type),
        )

        if s_attn:
            self.s_attention = SpatialAttention(kernel_size)

        if stride > 1:
            self.pool_layer = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if in_channels != out_channels:
            self.adjust_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                NormLayer3D(out_channels, norm_type),
            )

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)

        if hasattr(self, "c_attention"):
            out = self.c_attention(out)

        out = self.pointwise_conv2(out)

        if hasattr(self, "s_attention"):
            out = self.s_attention(out)

        if hasattr(self, "pool_layer"):
            x = self.pool_layer(x)

        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        
        return F.relu(out + x)


class InvResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type, up_scale, sq_r, c_attn, s_attn):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels, inner_channels + 16, 2, norm_type, up_scale, sq_r, kernel_size=7, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels + 16, inner_channels + 32, 2, norm_type, up_scale, sq_r, kernel_size=5, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels + 32, inner_channels + 48, 2, norm_type, up_scale, sq_r, c_attn=c_attn, s_attn=False),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d(inner_channels + 48, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        out = self.conv3d(x)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    net = InvResNet3D(5, 128, 16, 3, 0.1, "bn", 6, 16)
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    print(out.shape)

    count_params(net, "net")