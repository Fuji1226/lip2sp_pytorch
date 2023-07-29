import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inner_channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_layers = nn.Sequential(
            nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = x
        out = self.conv_layers(x)
        return out + res


class ResTCDecoder(nn.Module):
    """
    残差結合を取り入れたdecoder
    """
    def __init__(
        self, cond_channels, out_channels, inner_channels, n_layers, kernel_size, dropout, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor

        self.interp_layer = nn.Conv1d(cond_channels, inner_channels, kernel_size=1)

        self.conv_layers = nn.ModuleList(
            ResBlock(inner_channels, kernel_size, dropout) for _ in range(n_layers)
        )

        self.out_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

    def forward(self, enc_output):
        """
        enc_outout : (B, T, C)
        """
        enc_output = enc_output.permute(0, -1, 1)   # (B, C, T)
        out = enc_output

        # 音響特徴量のフレームまでアップサンプリング
        out = F.interpolate(out ,scale_factor=self.reduction_factor, mode="nearest")
        out = self.interp_layer(out)

        for layer in self.conv_layers:
            out = layer(out)

        out = self.out_layer(out)
        return out


class TConvBlock(nn.Module):
    def __init__(self, inner_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class WaveformDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.first_layer = nn.Conv1d(in_channels, inner_channels, kernel_size=1)

        self.mid_layers = nn.Sequential(
            TConvBlock(inner_channels, kernel_size=10, stride=5),
            TConvBlock(inner_channels, kernel_size=4, stride=2),
            TConvBlock(inner_channels, kernel_size=4, stride=2),
            TConvBlock(inner_channels, kernel_size=4, stride=2),
            TConvBlock(inner_channels, kernel_size=4, stride=2),
            TConvBlock(inner_channels, kernel_size=4, stride=2),
        )

        self.last_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        B, C, T = x.shape
        out = self.first_layer(x)
        out = self.mid_layers(out) 
        out = out.unsqueeze(2)  # (B, C, 1, T)

        # サンプル数を完全に合わせるのが難しかったので,最後は線形補間で対応
        out = F.interpolate(out, (1, int(T * self.scale_factor)), mode="nearest").squeeze(2)
        out = self.last_layer(out)
        return out


class LinearDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.layer = nn.Linear(in_channels, out_channels * reduction_factor)

    def forward(self, x):
        '''
        x : (B, T, C)
        '''
        x = self.layer(x)
        B, T, C = x.shape
        x = x.reshape(B, T * self.reduction_factor, C // self.reduction_factor)
        x = x.permute(0, 2, 1)  # (B, C, T)
        return x