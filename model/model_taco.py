import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D
from model.pre_post import Postnet
from model.rnn import LSTMEncoder
from model.rnn_decoder import RNNDecoder


class Lip2SPTaco(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels, norm_type,
        which_res, which_encoder,
        d_model, n_layers, n_head,
        rnn_hidden_channels, rnn_n_layers,
        dec_n_layers, dec_hidden_channels, dec_conv_channels, dec_conv_kernel_size, dec_use_attention,
        n_speaker, spk_emb_dim,
        pre_inner_channels, post_inner_channels, post_n_layers, post_kernel_size,
        dec_dropout, res_dropout, reduction_factor=2):
        super().__init__()
        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
            )
        
        # encoder
        self.encoder = LSTMEncoder(
            hidden_channels=rnn_hidden_channels,
            n_layers=rnn_n_layers,
            bidirectional=True,
            dropout=res_dropout,
            reduction_factor=reduction_factor,
        )

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)
        self.spk_emb_layer = nn.Linear(rnn_hidden_channels + spk_emb_dim, rnn_hidden_channels)

        # decoder
        self.decoder = RNNDecoder(
            hidden_channels=dec_hidden_channels,
            out_channels=out_channels,
            reduction_factor=reduction_factor,
            pre_in_channels=int(out_channels * reduction_factor),
            pre_inner_channels=pre_inner_channels,
            dropout=dec_dropout,
            enc_channels=rnn_hidden_channels,
            n_layers=dec_n_layers,
            conv_channels=dec_conv_channels,
            conv_kernel_size=dec_conv_kernel_size,
            use_attention=dec_use_attention,
        )

        # postnet
        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_kernel_size, post_n_layers)

    def forward(self, lip, data_len, target=None, gc=None, training_method=None, mixing_prob=None):
        # resnet
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 

        # speaker embedding
        if gc is not None:
            spk_emb = self.emb_layer(gc)    # (B, C)
            spk_emb = spk_emb.unsqueeze(1).expand(-1, enc_output.shape[1], -1)
            enc_output = torch.cat([enc_output, spk_emb], dim=-1)
            enc_output = self.spk_emb_layer(enc_output)
        else:
            spk_emb = None

        # decoder
        dec_output, att_w = self.decoder(enc_output, data_len, target, training_method, mixing_prob)

        # postnet
        out = self.postnet(dec_output)
        return out, dec_output, att_w
