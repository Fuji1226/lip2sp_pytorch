import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging


class UpSampleNet(nn.Module):
    def __init__(self, upsample_scales):
        super().__init__()
        self.upsample_scales = upsample_scales

        convs = []
        for scale in upsample_scales:
            kernel_size = (1, scale * 2 + 1)
            padding = (0, (kernel_size[1] - 1) // 2)
            convs.append(
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
                    # nn.BatchNorm2d(1),
                    # nn.ReLU(),
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = x.unsqueeze(1)
        for scale, layer in zip(self.upsample_scales, self.convs):
            x = F.interpolate(x, scale_factor=(1, scale), mode="nearest")
            x = layer(x)
        return x.squeeze(1)


class ConvinUpSampleNet(nn.Module):
    def __init__(self, in_channels, upsample_scales):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm1d(in_channels),
            # nn.ReLU(),   
        )
        self.upsample_layers = UpSampleNet(upsample_scales)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.upsample_layers(x)
        return x


class WaveNetResBlock(nn.Module):
    def __init__(self, inner_channels, cond_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(inner_channels, int(inner_channels * 2), kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.cond_layer = nn.Conv1d(cond_channels, int(inner_channels * 2), kernel_size=1)
        self.out_layer = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        self.skip_layer = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(inner_channels)

    def forward(self, x, c):
        """
        x, c : (B, C, T)
        """
        out = self.dropout(x)
        out = self.conv(out)
        out1, out2 = torch.split(out, out.shape[1] // 2, dim=1)

        c = self.cond_layer(c)
        c1, c2 = torch.split(c, c.shape[1] // 2, dim=1)
        out1 = out1 + c1
        out2 = out2 + c2
        out = torch.tanh(out1) * torch.sigmoid(out2)

        skip_out = self.skip_layer(out)
        out = (self.out_layer(out) + x) * math.sqrt(0.5)
        # out = self.bn(out)

        return out, skip_out


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, cond_channels, upsample_scales, n_layers, n_stacks, dropout, kernel_size, use_weight_norm):
        super().__init__()
        layers_per_stack = n_layers // n_stacks
        self.first_conv = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        self.cond_upsample_layers = ConvinUpSampleNet(cond_channels, upsample_scales)

        convs = []
        for i in range(n_layers):
            dilation = 2 ** (i % layers_per_stack)
            convs.append(
                WaveNetResBlock(
                    inner_channels=inner_channels,
                    cond_channels=cond_channels,
                    kernel_size=kernel_size,
                    dilation=dilation, 
                    dropout=dropout,
                )
            )
        self.convs = nn.ModuleList(convs)

        self.out_layers = nn.Sequential(
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Conv1d(inner_channels, inner_channels, kernel_size=1),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Conv1d(inner_channels, out_channels, kernel_size=1)
        )

        if use_weight_norm:
            self.apply_weight_norm()
    
    def forward(self, x, c):
        """
        x, c : (B, C, T)
        """
        c = self.cond_upsample_layers(c)
        assert c.shape[-1] == x.shape[-1]

        x = self.first_conv(x)
        skips = 0
        for layer in self.convs:
            x, skip = layer(x, c)
            skips += skip
        skips *= math.sqrt(1.0 / len(self.convs))

        x = skips
        x = self.out_layers(x)

        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


if __name__ == "__main__":
    net = Generator(
        in_channels=1,
        out_channels=1,
        inner_channels=64,
        cond_channels=80,
        upsample_scales=[10, 4, 2, 2],
        n_layers=30,
        n_stacks=3,
        dropout=0.1,
        kernel_size=3,
    )
    x = torch.randn(1, 1, 16000)
    c = torch.rand(1, 80, x.shape[-1] // 160)
    out = net(x, c)
    print(out.shape)