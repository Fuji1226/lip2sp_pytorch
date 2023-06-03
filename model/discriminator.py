"""
discriminatorの実装
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np


class MelDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels_list = [1, 32, 64, 128]
        out_channels_list = [32, 64, 128, 256]
        for i, (in_channels, out_channels) in enumerate(zip(in_channels_list, out_channels_list)):
            if i <= 1:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.MaxPool2d(3, 2, 1)
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                ))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)
        x = torch.mean(x, dim=(2, 3))   # (B, C)
        return x, fmaps


class AEDiscriminator(nn.Module):
    def __init__(self, in_channels, n_frame=150, hidden_channels=128, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(hidden_channels * n_frame // 2, 1),
        )

    def forward(self, x):
        """
        x : (B, T, C)
        out : (B, C)
        """
        return self.layers(x.permute(0, 2, 1))


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
        ])

        self.out_layer = nn.Linear(128, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = x.permute(0, -1, 1)     # (B, T, C)
        x = self.dropout(x)

        fmaps = []
        out = x

        for layer in self.layers:
            out = layer(out)
            fmaps.append(out)

        out = F.relu(self.out_layer(out))
        return [out], fmaps


