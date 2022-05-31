"""
lip2sp/links/cnn.pyの再現
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import chainer
import chainer.links as L
import chainer.functions as cF

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

class FrontEnd(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, 3, stride=(2, 2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, out_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        return x

class ChainerFrontEnd(chainer.Chain):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        with self.init_scope():
            self.conv1 = L.Convolution3D(None, 32, 3, (2, 2, 1), pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution3D(32, 32, 3, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution3D(32, out_channels, 3, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = cF.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = cF.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = cF.relu(x)
        x = cF.max_pooling_nd(x, ksize=(3, 3, 1), stride=(2, 2, 1), pad=(1, 1, 0))

        return x

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
class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, layers=5) -> None:
        super().__init__()

        self.frontend = FrontEnd(in_channels, out_channels)

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
        # print(f'after GAP {h.shape}')
        return h



if __name__ == "__main__":
    B = 4
    C = 5
    W = 48
    H = 48
    T = 150

    # W, Hが小さいと途中でカーネルサイズより小さくなっちゃって通りませんでした
    # とりあえず上の条件で一応通ると思います
    x = torch.rand(B, C, W, H, T)

    #x = np.random.rand(B, C, W, H, T)
    #x = chainer.Variable(x)

    net = ResNet3D(C, 64)
    out = net(x)    # (B, C, T=150)
    print(out.shape)

