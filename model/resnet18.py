import torch
import torch.nn as nn
import torch.nn.functional as F


class Block1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=(stride, stride, 1)),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        if stride > 1:
            self.pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0))

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
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(7, 7, 5), stride=(2, 2, 1), padding=(3, 3, 2)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.conv2 = nn.Sequential(
            Block1(hidden_channels, hidden_channels),
            Block1(hidden_channels, hidden_channels),
        )

        self.conv3 = nn.Sequential(
            Block2(hidden_channels, hidden_channels * 2),
            Block2(hidden_channels * 2, hidden_channels * 4),
            Block2(hidden_channels * 4, hidden_channels * 8),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.conv3(out)

        out = torch.mean(out, dim=(2, 3))
        return out


def check_params(net):
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")


if __name__ == "__main__":
    net = ResNet18(5, 512, 32)
    check_params(net)