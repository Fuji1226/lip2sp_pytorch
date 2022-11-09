import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F


class NormLayer3D(nn.Module):
    def __init__(self, in_channels, norm_type):
        super().__init__()
        self.norm_type = norm_type
        self.b_n = nn.BatchNorm3d(in_channels)
        self.i_n = nn.InstanceNorm3d(in_channels)

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        if self.norm_type == "bn":
            out = self.b_n(x)
        elif self.norm_type == "in":
            out = self.i_n(x)
        return out


class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1, stride, stride), padding=1),
            NormLayer3D(out_channels, norm_type),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        return self.layers(x)


class MultiDilatedConv(nn.Module):
    def __init__(self, inner_channels, stride, norm_type, n_groups):
        super().__init__()
        assert inner_channels % n_groups == 0
        self.split_channels = inner_channels // n_groups

        layers = []
        for g in range(n_groups):
            dilation = g + 1
            padding = dilation
            layers.append(
                nn.Sequential(
                    nn.Conv3d(self.split_channels, self.split_channels, kernel_size=3, stride=(1, stride, stride), dilation=(dilation, 1, 1), padding=(padding, 1, 1)),
                    NormLayer3D(self.split_channels, norm_type),
                    nn.ReLU(),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x_split = torch.split(x, self.split_channels, dim=1)

        out = []
        for each_x, layer in zip(x_split, self.layers):
            out.append(layer(each_x))
        
        return torch.cat(out, dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            NormalConv(in_channels, out_channels, stride, norm_type),
            NormalConv(out_channels, out_channels, 1, norm_type),
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
    def __init__(self, in_channels, out_channels, stride, norm_type, n_groups):
        super().__init__()
        self.layers = nn.Sequential(
            NormalConv(in_channels, out_channels, stride, norm_type),
            MultiDilatedConv(out_channels, 1, norm_type, n_groups),
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
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            ResBlock(inner_channels, inner_channels * 2, 2, norm_type),            
            nn.Dropout(dropout),

            ResBlock(inner_channels * 2, inner_channels * 4, 2, norm_type),
            nn.Dropout(dropout),
            
            ResBlock(inner_channels * 4, inner_channels * 8, 2, norm_type),
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
    def __init__(self, in_channels, out_channels, inner_channels, dropout, norm_type, n_groups):
        super().__init__()
        self.conv3d = nn.Sequential(
            NormalConv(in_channels, inner_channels, 2, norm_type),
            nn.Dropout(dropout),

            MultiDilatedResBlock(inner_channels, inner_channels * 2, 2, norm_type, n_groups),            
            nn.Dropout(dropout),

            MultiDilatedResBlock(inner_channels * 2, inner_channels * 4, 2, norm_type, n_groups),
            nn.Dropout(dropout),
            
            MultiDilatedResBlock(inner_channels * 4, inner_channels * 8, 2, norm_type, n_groups),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        out = self.conv3d(x)
        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)   # (B, C, T)
        return out


if __name__ == "__main__":
    net = MultiDilatedResNet3D(
        in_channels=3,
        out_channels=256,
        inner_channels=32,
        dropout=0.1,
        norm_type="bn",
        n_groups=4,
    )
    x = torch.rand(1, 3, 48, 48, 150)
    out = net(x)
    print(out.shape)