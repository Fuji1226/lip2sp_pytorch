"""
最終的なモデル
"""

from numpy import inner
import torch
import torch.nn as nn
from net import ResNet3D
from transformer import Prenet, Postnet, Encoder, Decoder
from glu import GLU


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        d_model, n_layers, n_head, d_k, d_v, d_inner,
        pre_in_channels, pre_inner_channels, pre_out_channels, post_inner_channels):
        super().__init__()

        self.first_batch_norm = nn.BatchNorm3d(in_channels)
        self.ResNet_GAP = ResNet3D(in_channels, d_model, layers=5)

        self.transformer_encoder = Encoder(
            n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, n_position=150)

        self.transformer_decoder = Decoder(
            n_layers, n_head, d_k, d_v, d_model, d_inner, 
            pre_in_channels, pre_inner_channels, pre_out_channels, 
            out_channels, use_gc=False, dropout=0.1, n_position=150, reduction_factor=2)

        self.postnet = Postnet(out_channels, post_inner_channels, out_channels)

    def forward(self):
        return
