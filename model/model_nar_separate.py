import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D, MultiDilatedResNet3D, AttentionResNet3D
from model.nar_decoder import ResTCDecoder
from model.rnn import GRUEncoder
from model.transformer_remake import Encoder
from model.grad_reversal import GradientReversal
from model.classifier import SpeakerClassifier


class Encoder(nn.Module):
    def __init__(
        self, in_channels, res_inner_channels, which_res,
        rnn_n_layers, trans_n_layers, trans_n_head,
        which_encoder, res_dropout, rnn_dropout, reduction_factor,
        n_speaker, spk_emb_dim):
        super().__init__()

        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=int(res_inner_channels * 8), 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
            )

        if which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=trans_n_layers, 
                n_head=trans_n_head, 
                d_model=int(res_inner_channels * 8), 
                reduction_factor=reduction_factor,  
            )
        elif which_encoder == "gru":
            self.encoder = GRUEncoder(
                hidden_channels=int(res_inner_channels * 8),
                n_layers=rnn_n_layers,
                bidirectional=True,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
            )

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)
        self.spk_emb_layer = nn.Conv1d(int(res_inner_channels * 8) + spk_emb_dim, int(res_inner_channels * 8), kernel_size=1)

    def forward(self, lip, data_len=None):
        lip_feature = self.ResNet_GAP(lip)
        enc_output = self.encoder(lip_feature, data_len)
        return enc_output

    
class MelDecoder(nn.Module):
    def __init__(
        self, res_inner_channels, out_channels,
        dec_n_layers, dec_kernel_size, dec_dropout, reduction_factor):
        super().__init__()
        self.decoder = ResTCDecoder(
            cond_channels=int(res_inner_channels * 8),
            out_channels=out_channels,
            inner_channels=int(res_inner_channels * 8),
            n_layers=dec_n_layers,
            kernel_size=dec_kernel_size,
            dropout=dec_dropout,
            reduction_factor=reduction_factor,
        )

    def forward(self, enc_output):
        return self.decoder(enc_output)


class TransposedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, 2, 2), padding=(0, padding, padding)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        """
        return self.layer(x)


class VideoDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            TransposedConvBlock(in_channels, in_channels // 2, 4),
            TransposedConvBlock(in_channels // 2, in_channels // 4, 4),
            TransposedConvBlock(in_channels // 4, in_channels // 8, 4),
            TransposedConvBlock(in_channels // 8, in_channels // 16, 4),
        )
        self.out_layer = nn.Conv3d(in_channels // 16, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, T, C)
        """
        x = x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, C, T, H, W)
        x = F.interpolate(x, size=(x.shape[2], 3, 3))
        x = self.layers(x)
        x = self.out_layer(x).permute(0, 1, 3, 4, 2)    # (B, C, H, W, T)
        return x