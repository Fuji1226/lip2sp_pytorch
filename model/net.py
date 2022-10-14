import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F

from utils import count_params

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


class FrontEnd(nn.Module):
    """
    3層の3次元畳み込み
    空間方向に1/4まで圧縮する
    """
    def __init__(self, in_channels, out_channels, dropout, norm_type):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn1 = NormLayer3D(32, norm_type)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = NormLayer3D(32, norm_type)
        self.conv3 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = NormLayer3D(out_channels, norm_type)

        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        out : (B, C, T, H, W)
        (H, W)は1/4になります
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
    """
    3次元畳み込み2層を含んだ残差結合ブロック
    """
    def __init__(self, in_channels, out_channels, stride, dropout, norm_type):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=(1, stride, stride), padding=1, bias=False
        )
        self.bn1 = NormLayer3D(out_channels, norm_type)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = NormLayer3D(out_channels, norm_type)

        if stride > 1 or in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            self.res_bn = NormLayer3D(out_channels, norm_type)

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
            if self.stride > 1:
                y2 = F.avg_pool3d(y2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            y2 = self.res_bn(self.res_conv(y2))

        return F.relu(y1 + y2)


class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type) -> None:
        super().__init__()
        # FrondEndを通した時点で(H, W)が(48, 48) -> (12, 12)になるので,層数を制限しています
        assert layers <= 3

        self.frontend = FrontEnd(in_channels, inner_channels, dropout, norm_type)

        res_blocks = []
        res_blocks.append(ResidualBlock3D(
            inner_channels, inner_channels, stride=1, dropout=dropout, norm_type=norm_type
        ))

        # stride=2にすることで空間方向に圧縮する3次元畳み込み
        for _ in range(layers - 1):
            res_blocks.append(ResidualBlock3D(
                inner_channels, inner_channels, stride=2, dropout=dropout, norm_type=norm_type
            ))
        self.res_layers = nn.ModuleList(res_blocks)

        self.out_layer = nn.Conv3d(inner_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x : (B, C, H, W, T)
        out : (B, C, T)
        """
        # 3D convolution & MaxPooling
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        out = self.frontend(x)
        
        # residual layers
        for layer in self.res_layers:
            out = layer(out)

        out = self.out_layer(out)

        # W, HについてAverage pooling
        out = torch.mean(out, dim=(3, 4))
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            NormalConv(in_channels, out_channels, stride, norm_type),
            # NormalConv(out_channels, out_channels, 1, norm_type),
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


class Simple(nn.Module):
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
        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    net = Simple(5, 128, 32, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    count_params(net, "net")