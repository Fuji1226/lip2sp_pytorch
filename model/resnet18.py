import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_params

class Block1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, stride, stride)),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        if stride > 1:
            self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1))

        if in_channels != out_channels:
            self.adjust_layer = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)

        return out + x


class Block2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.layer1 = Block1(in_channels, out_channels, stride)
        self.layer2 = Block1(out_channels, out_channels, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = nn.Sequential(
            Block1(hidden_channels, hidden_channels),
            nn.Dropout(dropout),
            Block1(hidden_channels, hidden_channels),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            Block2(hidden_channels, hidden_channels * 2),
            nn.Dropout(dropout),
            Block2(hidden_channels * 2, hidden_channels * 4),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        x = x.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.conv3(out)

        out = torch.mean(out, dim=(3, 4))
        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    net = ResNet18(5, 128, 64)
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    count_params(net, "net")
    