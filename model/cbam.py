from tabnanny import check
import torch
import torch.nn as nn


class ChannnelAttention(nn.Module):
    def __init__(self, hidden_channels, sq_r):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // sq_r, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // sq_r, hidden_channels, 1, bias=False),
        )

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        B, C, H, W, T = x.shape
        out = x.permute(0, -1, 1, 2, 3)
        out = out.reshape(B * T, C, H, W)

        c_avg = self.channel_attention(self.avg_pool(out))
        c_max = self.channel_attention(self.max_pool(out))
        c_uni = torch.sigmoid(c_avg + c_max)
        c_uni = c_uni.reshape(B, T, C, 1, 1).permute(0, 2, 3, 4, 1)
        return x * c_uni


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, 3), padding=(padding, padding, 1), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        s_avg = torch.mean(x, dim=1, keepdim=True)
        s_max, _ = torch.max(x, dim=1, keepdim=True)
        s_uni = torch.cat([s_avg, s_max], dim=1)
        s_uni = self.spatial_attention(s_uni)
        return x * s_uni


class SpatialAttentionFC(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=(kernel_size, kernel_size, 1), padding=(padding, padding, 0), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        attn = self.spatial_attention(x)
        return x * attn


class CbamBlock(nn.Module):
    def __init__(self, hidden_channels, sq_r, kernel_size):
        super().__init__()
        self.c_attn = ChannnelAttention(hidden_channels, sq_r)
        self.s_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.c_attn(x)
        out = self.s_attn(out)
        return out


def check_params(net):
    x = torch.rand(1, 256, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")


if __name__ == "__main__":
    kernel_size = 7
    net = SpatialAttention(kernel_size)
    check_params(net)

    net = SpatialAttentionFC(256, kernel_size)
    check_params(net)

