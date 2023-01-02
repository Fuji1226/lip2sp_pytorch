import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, n_layers, kernel_size, use_weight_norm):
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
                    nn.Dropout(0.1),
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