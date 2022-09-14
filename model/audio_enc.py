import sys
from pathlib import Path
sys.path.append(Path("~/lip2sp_pytorch").expanduser())

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_remake import Encoder

# try:
#     from model.transformer_remake import Encoder
# except:
#     from .transformer_remake import Encoder

class NormLayer1D(nn.Module):
    def __init__(self, in_channels, norm_type):
        super().__init__()
        self.norm_type = norm_type
        self.b_n = nn.BatchNorm1d(in_channels)
        self.i_n = nn.InstanceNorm1d(in_channels)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        if self.norm_type == "bn":
            out = self.b_n(x)
        elif self.norm_type == "in":
            out = self.i_n(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            NormLayer1D(out_channels, norm_type),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            NormLayer1D(out_channels, norm_type),
            nn.ReLU(),
        )

    def forward(self, x):
        res = x
        out = self.layers(x)
        out += res
        return out


class ContentEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_attn_layer, n_head, reduction_factor, norm_type):
        super().__init__()
        assert out_channels % n_head == 0
        
        self.first_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_layers = nn.ModuleList([
            ResBlock(out_channels, out_channels, kernel_size=3, norm_type=norm_type),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            NormLayer1D(out_channels, norm_type),
            nn.ReLU(),
            ResBlock(out_channels, out_channels, kernel_size=3, norm_type=norm_type),
        ])

        self.attention = Encoder(
            n_layers=n_attn_layer, 
            n_head=n_head, 
            d_model=out_channels, 
            reduction_factor=reduction_factor,  
        )

    def forward(self, x, data_len=None):
        """
        x : (B, C, T)
        out : (B, T, C)
        """
        out = self.first_conv(x)
        for layer in self.conv_layers:
            out = layer(out)
        out = self.attention(out, data_len)
        return out


class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        o = out_channels
        in_cs = [in_channels, o, o, o, o]
        out_cs = [o, o, o, o, o]
        stride = [2, 1, 2, 1, 2]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, stride=s, padding=1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
            ) for in_c, out_c, s in zip(in_cs, out_cs, stride)
        ])
        self.last_conv = nn.Conv1d(o, o, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(o)

    def forward(self, x):
        """
        x : (B, C, T)
        out : (B, C)
        """
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.last_conv(out)
        # out = self.bn(out)
        
        # average pooling
        out = torch.mean(out, dim=-1)
        return out


if __name__ == "__main__":
    con_enc = ContentEncoder(
        in_channels=80,
        out_channels=256,
        n_attn_layer=1,
        n_head=4,
        reduction_factor=2,
    )
    spk_enc = SpeakerEncoder(
        in_channels=80,
        out_channels=256,
    )
    x = torch.rand(8, 80, 300)

    con_out = con_enc(x)
    print(f"con_out = {con_out.shape}")

    spk_out = spk_enc(x)
    print(f"spk_out = {spk_out.shape}")