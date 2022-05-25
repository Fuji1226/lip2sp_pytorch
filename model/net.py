"""
lip2sp/links/cnn.pyの再現
"""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_frontend(in_channels, out_channels):
    frontend = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=(2, 2, 1), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 32, 3, stride=(2, 2, 1), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, out_channels, 3, stride=(2, 2, 1), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
        )
    return frontend


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
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
            self.conv4 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
            self.bn4 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y1 = F.relu(y1)
        y1 = self.conv2(y1)
        y1 = self.bn2(y1)

        y2 = x
        if hasattr(self, "conv4"):
            y2 = F.avg_pool3d(y2, (3, 3, 1), (2, 2, 1), (1, 1, 0))
            y2 = self.bn4(self.conv4(y2))

        return F.relu(y1 + y2)




####３DCNN+GAP#####
class PreNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers=5) -> None:
        super().__init__()

        self.frontend = build_frontend(in_channels, out_channels)

        res_blocks = nn.Sequential()
        res_blocks.append(ResidualBlock3D(
            out_channels, out_channels, stride=1,
        ))

        for _ in range(min(layers, 2)):
            res_blocks.append(ResidualBlock3D(
                out_channels, out_channels, stride=2,
            ))
        self.res_block = res_blocks
    
    def forward(self, x):
        """
        x : (Batch, Channel, Width, Height, Time)

        return
        h : (B, C, T)
        """
        h = self.frontend(x)
        h = self.res_block(h)
        # W, HについてAverage pooling
        h = torch.mean(h, dim=(2, 3))
        return h


if __name__ == "__main__":
    B = 1
    C = 3
    W = 128
    H = 128
    T = 100

    # W, Hが小さいと途中でカーネルサイズより小さくなっちゃって通りませんでした
    # とりあえず上の条件で一応通ると思います
    x = torch.rand(B, C, W, H, T)
    net = PreNet(3, 128)
    out = net(x)
    print(out.size())

