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
        x : (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        out = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        out = out.reshape(B * T, C, H, W)

        c_avg = self.channel_attention(self.avg_pool(out))
        c_max = self.channel_attention(self.max_pool(out))
        c_uni = torch.sigmoid(c_avg + c_max)
        c_uni = c_uni.reshape(B, T, C, 1, 1).permute(0, 2, 1, 3, 4)     # (B, C, T, 1, 1)
        return x * c_uni


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(3, kernel_size, kernel_size), padding=(1, padding, padding), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        s_avg = torch.mean(x, dim=1, keepdim=True)  # (B, 1, T, H, W)
        s_max, _ = torch.max(x, dim=1, keepdim=True)    # (B, 1, T, H, W)
        s_uni = torch.cat([s_avg, s_max], dim=1)    # (B, 2, T, H, W)
        s_uni = self.spatial_attention(s_uni)
        return x * s_uni

