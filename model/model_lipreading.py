"""
Lipreadingに使用する予定のモデル
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder, PhonemeDecoder
    from model.conformer.encoder import ConformerEncoder
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, PhonemeDecoder
    from .conformer.encoder import ConformerEncoder


def token_mask(x):
    """
    音素ラベルに対してのマスクを作成
    MASK_INDEXに一致するところをマスクする

    x : (B, T)
    mask : (B, T)
    """
    MASK_INDEX = 0
    zero_matrix = torch.zeros_like(x)
    one_matrix = torch.ones_like(x)
    mask = torch.where(x == MASK_INDEX, one_matrix, zero_matrix).bool() 
    return mask


class Lip2Text(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels,
        trans_enc_n_layers, trans_enc_n_head,
        res_dropout, reduction_factor):
        super().__init__()
        self.out_channels = out_channels

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=int(res_inner_channels * 8), 
            inner_channels=res_inner_channels,
            dropout=res_dropout,
            layers=3,
            norm_type='bn'
        )
        inner_channels = int(res_inner_channels * 8)

        # encoder
        self.encoder = Encoder(
            n_layers=trans_enc_n_layers, 
            n_head=trans_enc_n_head, 
            d_model=inner_channels, 
            reduction_factor=reduction_factor,  
        )

        self.ctc_output_layer = nn.Linear(inner_channels, 52)

    def forward(self, lip, data_len):
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化

        # encoder
        enc_output = self.ResNet_GAP(lip)
        enc_output = self.encoder(enc_output, data_len)    # (B, T, C)
        ctc_output = self.ctc_output_layer(enc_output)  # (B, T, C)
        
        output = {}
        output['ctc_output'] = ctc_output
        return output
        