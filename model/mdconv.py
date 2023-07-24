"""
MixDepthwiseConv
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
from dsconv import DSLayer3D
from utils import count_params


class MDConv(nn.Module):
    def __init__(self, inner_channels, n_groups, stride):
        super().__init__()
        self.split_channels = inner_channels // n_groups
        layers = []
        for i in range(n_groups):
            kernel_size = int(2 * i + 3)
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv3d(self.split_channels, self.split_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, padding, padding), groups=self.split_channels)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        # チャンネル方向に分割
        x_split = torch.split(x, self.split_channels, dim=1)

        # それぞれのグループに対して異なるカーネルサイズで空間方向に畳み込みを適用し、結合して戻す
        out = torch.cat([layer(input) for input, layer in zip(x_split, self.layers)], dim=1)
        return out


class InvResLayerMD(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, stride, norm_type, up_scale, sq_r, kernel_size=None, c_attn=True, s_attn=True):
        super().__init__()
        self.hidden_channels = int(in_channels * up_scale)
        self.split_channels = self.hidden_channels // n_groups
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            NormLayer3D(self.hidden_channels, norm_type),
            nn.ReLU(),
        )
        self.depthwise_conv = nn.Sequential(
            MDConv(self.hidden_channels, n_groups, stride),
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


class InvResNetMD(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type, n_groups, up_scale, sq_r, c_attn, s_attn, n_add_channels):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            InvResLayerMD(inner_channels, inner_channels + n_add_channels, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=7, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # spatial attentionに入力される特徴マップが6×6なので、kernel_sizeは7でなく5にする
            InvResLayerMD(inner_channels + n_add_channels, inner_channels + n_add_channels * 2, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=5, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # 最後は6×6で入力され、特徴マップがかなり小さいのでMix Convolutionは使用しない通常のInverted Residual Block
            InvResLayer3D(inner_channels + n_add_channels * 2, inner_channels + n_add_channels * 3, 2, norm_type, up_scale, sq_r, c_attn=c_attn, s_attn=False),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d(inner_channels + n_add_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        out = self.conv3d(x)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)
        return out


class InvResNetMD_DSOut(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type, n_groups, up_scale, sq_r, c_attn, s_attn, n_add_channels):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            InvResLayerMD(inner_channels, inner_channels + n_add_channels, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=7, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # spatial attentionに入力される特徴マップが6×6なので、kernel_sizeは7でなく5にする
            InvResLayerMD(inner_channels + n_add_channels, inner_channels + n_add_channels * 2, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=5, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # 最後は6×6で入力され、特徴マップがかなり小さいのでMix Convolutionは使用せず、通常のDepthwise & Pointwise Convolutionを適用
            DSLayer3D(inner_channels + n_add_channels * 2, (inner_channels + n_add_channels * 2) * 2, 2, norm_type),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d((inner_channels + n_add_channels * 2) * 2, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        out = self.conv3d(x)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)
        return out


class InvResNetMDBig(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type, n_groups, up_scale, sq_r, c_attn, s_attn):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            InvResLayerMD(inner_channels, inner_channels, n_groups, 1, norm_type, up_scale, sq_r, kernel_size=7, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            InvResLayerMD(inner_channels, inner_channels + 16, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=7, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # spatial attentionに入力される特徴マップが6×6なので、kernel_sizeは7でなく5にする
            InvResLayerMD(inner_channels + 16, inner_channels + 16, n_groups, 1, norm_type, up_scale, sq_r, kernel_size=5, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # spatial attentionに入力される特徴マップが6×6なので、kernel_sizeは7でなく5にする
            InvResLayerMD(inner_channels + 16, inner_channels + 32, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=5, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # 最後は6×6で入力され、特徴マップがかなり小さいのでMix Convolutionは使用しない通常のInverted Residual Block
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


class InvResLayerMD_NonRes(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, stride, norm_type, up_scale, sq_r, kernel_size=None, c_attn=True, s_attn=True):
        super().__init__()
        self.hidden_channels = int(in_channels * up_scale)
        self.split_channels = self.hidden_channels // n_groups
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            NormLayer3D(self.hidden_channels, norm_type),
            nn.ReLU(),
        )
        self.depthwise_conv = nn.Sequential(
            MDConv(self.hidden_channels, n_groups, stride),
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

        return F.relu(out)


class InvResNetMD_DSOut_NonRes(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type, n_groups, up_scale, sq_r, c_attn, s_attn, n_add_channels):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            InvResLayerMD_NonRes(inner_channels, inner_channels + n_add_channels, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=7, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # spatial attentionに入力される特徴マップが6×6なので、kernel_sizeは7でなく5にする
            InvResLayerMD_NonRes(inner_channels + n_add_channels, inner_channels + n_add_channels * 2, n_groups, 2, norm_type, up_scale, sq_r, kernel_size=5, c_attn=c_attn, s_attn=s_attn),
            nn.Dropout(dropout),

            # 最後は6×6で入力され、特徴マップがかなり小さいのでMix Convolutionは使用せず、通常のDepthwise & Pointwise Convolutionを適用
            DSLayer3D(inner_channels + n_add_channels * 2, (inner_channels + n_add_channels * 2) * 2, 2, norm_type),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d((inner_channels + n_add_channels * 2) * 2, out_channels, kernel_size=1)

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
    inner_channels = 32
    n_add_channels = 32
    up_scale = 4
    net = InvResNetMD_DSOut_NonRes(5, 128, inner_channels, 3, 0.1, "bn", 4, up_scale, 16, True, True, n_add_channels)
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    print(out.shape)
    count_params(net, "net")
