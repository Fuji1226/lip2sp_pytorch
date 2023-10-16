from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_padding

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, inner_channels, kernel_size, dilation):
        super().__init__()
        convs1 = []
        convs2 = []
        for i in range(len(dilation)):
            convs1.append(
                nn.Sequential(
                    nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])),        
                    nn.BatchNorm1d(inner_channels),
                    nn.LeakyReLU(LRELU_SLOPE),
                )
            )
            convs2.append(
                nn.Sequential(
                    nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, dilation=1, padding=get_padding(kernel_size, 1)),    
                    nn.BatchNorm1d(inner_channels),
                    nn.LeakyReLU(LRELU_SLOPE),
                )
            )
        self.convs1 = nn.ModuleList(convs1)
        self.convs2 = nn.ModuleList(convs2)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(
        self, in_channels, upsample_initial_channels, upsample_rates, upsample_kernel_sizes,
        res_kernel_sizes, res_dilations):
        super().__init__()
        self.num_res_kernels = len(res_kernel_sizes)

        self.first_conv = nn.Conv1d(in_channels, upsample_initial_channels, kernel_size=7, padding=3)

        upsample_layers = []
        for i, (k, s) in enumerate(zip(upsample_kernel_sizes, upsample_rates)):
            in_c = upsample_initial_channels // (2 ** i)
            out_c = upsample_initial_channels // (2 ** (i + 1))
            padding = (k - s) // 2
            upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_c, out_c, kernel_size=k, stride=s, padding=padding),
                    nn.BatchNorm1d(out_c),
                    nn.LeakyReLU(LRELU_SLOPE),
                )
            )
        self.upsample_layers = nn.ModuleList(upsample_layers)

        res_blocks = []
        for i in range(len(self.upsample_layers)):
            c = upsample_initial_channels // (2 ** (i + 1))
            for k, d in zip(res_kernel_sizes, res_dilations):
                res_blocks.append(ResBlock(c, k, d))
        self.res_blocks = nn.ModuleList(res_blocks)

        last_c = upsample_initial_channels // (2 ** (len(self.upsample_layers)))
        self.out_layer = nn.Conv1d(last_c, 1, kernel_size=7, padding=3)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        out = self.first_conv(x)

        for i in range(len(self.upsample_layers)):
            out = self.upsample_layers[i](out)

            res_out = None
            for j in range(self.num_res_kernels):
                if res_out is None:
                    res_out = self.res_blocks[i * self.num_res_kernels + j](out)
                else:
                    res_out += self.res_blocks[i * self.num_res_kernels + j](out)
            out = res_out / self.num_res_kernels

        out = self.out_layer(out)
        out = torch.tanh(out)

        return out


if __name__ == "__main__":
    net = Generator(
        in_channels=80,
        upsample_initial_channels=128,
        upsample_rates=[10, 4, 2, 2],
        upsample_kernel_sizes=[20, 8, 4, 4],
        res_kernel_sizes=[3, 7, 11],
        res_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )
    x = torch.rand(1, 80, 100)
    out = net(x)
    print(out.shape)