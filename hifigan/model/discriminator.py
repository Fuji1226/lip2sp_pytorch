from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_padding
from hifigan.model.generator import LRELU_SLOPE


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size, stride):
        super().__init__()
        self.period = period
        in_cs = [1, 32, 128, 512, 1024]
        out_cs = [32, 128, 512, 1024, 1024]

        convs = []
        for i, (in_c, out_c) in enumerate(zip(in_cs, out_cs)):
            if i < len(in_cs) - 1:
                conv = nn.Conv2d(in_c, out_c, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))
            else:
                conv = nn.Conv2d(in_c, out_c, kernel_size=(kernel_size, 1), stride=1, padding=(get_padding(kernel_size, 1), 0))
            convs.append(
                nn.Sequential(
                    conv,
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(LRELU_SLOPE),
                )
            )
        self.convs = nn.ModuleList(convs)

        self.last_layer = nn.Conv2d(out_cs[-1], 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        """
        x : (B, C, T)
        """
        fmaps = []

        B, C, T = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            T = T + n_pad
        x = x.reshape(B, C, T // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            fmaps.append(x)

        x = self.last_layer(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, 5, 3),
            DiscriminatorP(3, 5, 3),
            DiscriminatorP(5, 5, 3),
            DiscriminatorP(7, 5, 3),
            DiscriminatorP(11, 5, 3),
        ])

    def forward(self, x, x_pred):
        out_real_list = []
        out_pred_list = []
        fmaps_real_list = []
        fmaps_pred_list = []

        for disc in self.discriminators:
            out_real, fmaps_real = disc(x)
            out_pred, fmaps_pred = disc(x_pred)
            out_real_list.append(out_real)
            out_pred_list.append(out_pred)
            fmaps_real_list.append(fmaps_real)
            fmaps_pred_list.append(fmaps_pred)

        return out_real_list, out_pred_list, fmaps_real_list, fmaps_pred_list


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        in_cs = [1, 128, 128, 256, 512, 1024, 1024]
        out_cs = [128, 128, 256, 512, 1024, 1024, 1024]
        kernel_sizes = [15, 41, 41, 41, 41, 41, 5]
        strides = [1, 2, 2, 4, 4, 1, 1]
        groups = [1, 4, 16, 16, 16, 16, 1]

        convs = []
        for in_c, out_c, k, s, g in zip(in_cs, out_cs, kernel_sizes, strides, groups):
            padding = get_padding(k, 1)
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, groups=g, padding=padding),
                    nn.BatchNorm1d(out_c),
                    nn.LeakyReLU(LRELU_SLOPE),
                )
            )
        self.convs = nn.ModuleList(convs)

        self.last_layer = nn.Conv1d(out_cs[-1], 1, kernel_size=3, padding=1)

    def forward(self, x):
        fmaps = []

        for layer in self.convs:
            x = layer(x)
            fmaps.append(x)

        x = self.last_layer(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.mean_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
        ])

    def forward(self, x, x_pred):
        out_real_list = []
        out_pred_list = []
        fmaps_real_list = []
        fmaps_pred_list = []

        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x = self.mean_pools[i - 1](x)
                x_pred = self.mean_pools[i - 1](x_pred)

            out_real, fmaps_real = disc(x)
            out_pred, fmaps_pred = disc(x_pred)
            out_real_list.append(out_real)
            out_pred_list.append(out_pred)
            fmaps_real_list.append(fmaps_real)
            fmaps_pred_list.append(fmaps_pred)

        return out_real_list, out_pred_list, fmaps_real_list, fmaps_pred_list


if __name__ == "__main__":
    net = MultiPeriodDiscriminator()
    x = torch.rand(1, 1, 300)
    x_pred = torch.rand_like(x)
    out_real_list, out_pred_list, fmaps_real_list, fmaps_pred_list = net(x, x_pred)
    
    net = MultiScaleDiscriminator()
    x = torch.rand(1, 1, 300)
    x_pred = torch.rand_like(x)
    out_real_list, out_pred_list, fmaps_real_list, fmaps_pred_list = net(x, x_pred)