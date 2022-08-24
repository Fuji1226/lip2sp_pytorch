"""
lip2sp/links/cnn.pyの再現
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrontEnd(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 7), stride=(2, 2, 1), padding=(1, 1, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        out : (B, C, H, W, T)
        (H, W)は1/4になります(stride=2が2層あるので) (48, 48) -> (12, 12)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=(stride, stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride > 1 or in_channels != out_channels:
            self.res_conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
            self.res_bn = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        """
        stride=2の場合,(H, W)は1/2になります
        """
        y1 = self.dropout(x)

        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = F.relu(y1)
            
        y1 = self.conv2(y1)
        y1 = self.bn2(y1)

        y2 = x
        if hasattr(self, "res_conv"):
            # 空間方向のstrideが2の場合、空間方向に1/2に圧縮されるのでその分を考慮
            y2 = F.avg_pool3d(y2, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
            y2 = self.res_bn(self.res_conv(y2))

        return F.relu(y1 + y2)
        

class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout) -> None:
        super().__init__()
        # FrondEndを通した時点で(H, W)が(48, 48) -> (12, 12)になるので,層数を制限しています
        assert layers <= 3

        self.frontend = FrontEnd(in_channels, inner_channels, dropout)

        res_blocks = []
        res_blocks.append(ResidualBlock3D(
            inner_channels, inner_channels, stride=1, dropout=dropout,
        ))

        # stride=2にすることで空間方向に圧縮する3次元畳み込み
        for _ in range(layers - 1):
            res_blocks.append(ResidualBlock3D(
                inner_channels, inner_channels, stride=2, dropout=dropout,
            ))
        self.res_layers = nn.ModuleList(res_blocks)

        self.out_layer = nn.Conv3d(inner_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x : (B, C, H, W, T)
        out : (B, C, T)
        """
        # 3D convolution & MaxPooling
        out = self.frontend(x)
        
        # residual layers
        for layer in self.res_layers:
            out = layer(out)

        out = self.out_layer(out)

        # W, HについてAverage pooling
        out = torch.mean(out, dim=(2, 3))
        return out
