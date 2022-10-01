import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        print(dilation)
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer(x)
        return out + x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** (i + 1)
            layers.append(
                DilatedConvBlock(in_channels, in_channels, kernel_size, dilation),
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, data_len=None):
        """
        x : (B, C, T)
        out : (B, T, C)
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.permute(0, 2, 1)


if __name__ == "__main__":
    net = DilatedConvEncoder(
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        n_layers=5,
    )
    x = torch.rand(1, 128, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")