"""
非自己回帰のモデル
GANをやるなら自己回帰はあってなさそうなので作成
"""
import os
import sys
from pathlib import Path
sys.path.append(Path("~/lip2sp_pytorch/model").expanduser())

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder
    from model.conformer.encoder import ConformerEncoder
    from model.nar_decoder import TCDecoder, GatedTCDecoder, ResTCDecoder
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder
    from .conformer.encoder import ConformerEncoder
    from .nar_decoder import TCDecoder, GatedTCDecoder, ResTCDecoder


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels,
        d_model, n_layers, n_head, conformer_conv_kernel_size,
        dec_n_layers, dec_inner_channels, dec_kernel_size,
        feat_add_channels, feat_add_layers,
        which_encoder, which_decoder, apply_first_bn, multi_task, add_feat_add,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False):
        super().__init__()

        assert d_model % n_head == 0
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        if apply_first_bn:
            self.first_batch_norm = nn.BatchNorm3d(in_channels)

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=d_model, 
            inner_channels=res_inner_channels,
            layers=res_layers, 
            dropout=res_dropout,
        )

        # encoder
        if self.which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=n_layers, 
                n_head=n_head, 
                d_model=d_model, 
                reduction_factor=reduction_factor,  
            )
        elif self.which_encoder == "conformer":
            self.encoder = ConformerEncoder(
                encoder_dim=d_model, 
                num_layers=n_layers, 
                num_attention_heads=n_head, 
                conv_kernel_size=conformer_conv_kernel_size,
                reduction_factor=reduction_factor,
            )

        # decoder
        if self.which_decoder == "simple_tc":
            self.decoder = TCDecoder(
                cond_channels=d_model,
                out_channels=out_channels,
                inner_channels=dec_inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
            )
        elif self.which_decoder == "gated_tc":
            self.decoder = GatedTCDecoder(
                cond_channels=d_model,
                out_channels=out_channels,
                inner_channels=dec_inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
            )
        elif self.which_decoder == "res_tc":
            self.decoder = ResTCDecoder(
                cond_channels=d_model,
                out_channels=out_channels,
                inner_channels=dec_inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
                feat_add_channels=feat_add_channels, 
                feat_add_layers=feat_add_layers,
                multi_task=multi_task,
                add_feat_add=add_feat_add,
            )

    def forward(self, lip, data_len=None, gc=None):
        # resnet
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 

        # decoder
        output = self.decoder(enc_output)
        return output
