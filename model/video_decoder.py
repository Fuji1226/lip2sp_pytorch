import torch
import torch.nn as nn
import torch.nn.functional as F


class TransposedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, 2, 2), padding=(0, padding, padding)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        return self.layer(x)


class VideoDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            TransposedConvBlock(in_channels, in_channels // 2, 4),
            TransposedConvBlock(in_channels // 2, in_channels // 4, 4),
            TransposedConvBlock(in_channels // 4, in_channels // 8, 4),
            TransposedConvBlock(in_channels // 8, in_channels // 16, 4),
        )
        self.out_layer = nn.Conv3d(in_channels // 16, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.interpolate(x, size=(x.shape[2], 3, 3))
        x = self.layers(x)
        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    B = 4
    C = 256
    T = 150
    HW = 48
    x = torch.rand(B, C, T)
    net = VideoDecoder(
        in_channels=256,
        out_channels=1,
    )
    out = net(x)
    print(out.shape)
