import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, n_layers, kernel_size, use_weight_norm, dropout):
        super().__init__()
        convs = []
        for i in range(n_layers - 1):
            if i == 0:
                dilation = 1
                conv_in_channels = in_channels
            else:
                dilation = i
                conv_in_channels = inner_channels

            padding = (kernel_size - 1) // 2 * dilation
            convs.append(
                nn.Sequential(
                    nn.Conv1d(conv_in_channels, inner_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                )
            )
        convs.append(
            nn.Conv1d(inner_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        self.convs = nn.ModuleList(convs)

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """
        x : (B, 1, T)
        """
        for layer in self.convs:
            x = layer(x)
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


class WaveNetResBlock(nn.Module):
    def __init__(self, inner_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.out_layer = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        self.skip_layer = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        out = self.dropout(x)
        out = self.conv(out)
        skip_out = self.skip_layer(out)
        out = (self.out_layer(out) + x) * math.sqrt(0.5)
        return out, skip_out


class WaveNetLikeDiscriminator(nn.Module):
    def __init__(self, n_layers, n_stacks, in_channels, inner_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.first_layer = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        layers_per_stack = n_layers // n_stacks

        convs = []
        for i in range(n_layers):
            dilation = 2 ** (i % layers_per_stack)
            convs.append(
                WaveNetResBlock(
                    inner_channels=inner_channels,
                    kernel_size=kernel_size,
                    dilation=dilation, 
                    dropout=dropout,
                )
            )
        self.convs = nn.ModuleList(convs)

        self.out_layers = nn.Sequential(
            nn.BatchNorm1d(inner_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(inner_channels, inner_channels, kernel_size=1),
            nn.BatchNorm1d(inner_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(inner_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.first_layer(x)

        skips = 0
        for layer in self.convs:
            x, skip = layer(x)
            skips += skip
        skips *= math.sqrt(1.0 / len(self.convs))
        x = skips
        x = self.out_layers(x)
        return x


if __name__ == "__main__":
    disc = WaveNetLikeDiscriminator(
        n_layers=30,
        layers_per_stack=10,
        in_channels=1,
        inner_channels=64,
        out_channels=1,
        kernel_size=3,
        dropout=0.1,
    )
    x = torch.rand(1, 1, 16000)
    output = disc(x)
    print(output.shape)

    disc = Discriminator(
        in_channels=1,
        out_channels=1,
        inner_channels=64,
        n_layers=10,
        kernel_size=3,
        use_weight_norm=False,
    )
    x = torch.rand(1, 1, 16000)
    output = disc(x)
    print(output.shape)