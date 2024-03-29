"""
glu
transformer_remake.pyと一緒に使えるようになっています
transformer_taguchi.pyとは使えません(データ形状が違うので)
"""
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_remake import shift
from model.pre_post import Prenet
from wavenet.model.conv import CausalConv1d   


class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropput):
        super().__init__()
        self.dropout = nn.Dropout(p=dropput)
        self.causal_conv = CausalConv1d(in_channels, out_channels * 2, kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, feature, x):
        return self._forward(feature, x, False)

    def incremental_forward(self, feature, x):
        return self._forward(feature, x, True)

    def _forward(self, enc_output, x, incremental):
        """
        enc_output（encoderからの特徴量） : (B, C, T)
        x（音響特徴量） : (B, C, T)

        incrementalは推論時に使用します
        """
        res = x
        y = self.dropout(x)
        
        if incremental:
            split_dim = 1
            y = self.causal_conv.incremental_forward(y)     # (B, C, T)
        else:
            split_dim = 1
            y = self.causal_conv(y)
        
        # チャンネル方向に2分割
        y1, y2 = torch.split(y, y.shape[split_dim] // 2, dim=split_dim)

        enc_output = F.softsign(self.conv(enc_output))
        y2 = y2 + enc_output
        out = torch.sigmoid(y1) * y2

        out += res
        out *= 0.5 ** 0.5
        return out

    def clear_buffer(self):
        self.causal_conv.clear_buffer()


class GLU(nn.Module):
    def __init__(
        self, inner_channels, out_channels, pre_in_channels, pre_inner_channels, cond_channels,
        reduction_factor, n_layers, kernel_size, dropout, use_spk_emb, spk_emb_dim):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        self.prenet = Prenet(pre_in_channels, inner_channels, pre_inner_channels)

        if use_spk_emb:
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        self.cond_layer = nn.Conv1d(cond_channels, inner_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.glu_layers = nn.ModuleList([
            GLUBlock(inner_channels, inner_channels, kernel_size, dropout) for _ in range(n_layers)
        ])
        self.conv_o = nn.Conv1d(inner_channels, self.out_channels * self.reduction_factor, kernel_size=1)

    def forward(self, enc_output, spk_emb, mode, prev=None):
        """
        enc_output : (B, T, C)
        spk_emb : (B, C)
        prev : (B, C, T)
        dec_output : (B, C, T)
        """
        assert mode == "training" or "inference"
        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)
        enc_output = self.cond_layer(enc_output)
        B = enc_output.shape[0]
        T = enc_output.shape[-1]
        D = self.out_channels

        # prev shift
        if mode == "training":
            prev = shift(prev, self.reduction_factor)

        # view for reduction factor
        if prev is not None:
            prev = prev.permute(0, -1, -2)  # (B, T, C)
            prev = prev.contiguous().view(B, -1, D * self.reduction_factor)
            prev = prev.permute(0, -1, -2)  # (B, C, T)
        else:
            prev = torch.zeros(B, D * self.reduction_factor, 1).to(device=enc_output.device, dtype=enc_output.dtype) 

        # prenet
        prev = self.dropout(self.prenet(prev))      # (B, C, T)

        if hasattr(self, "spk_emb_layer"):
            spk_emb = spk_emb.unsqueeze(-1).expand(-1, -1, prev.shape[-1])  # (B, C, T)
            prev = torch.cat([prev, spk_emb], dim=1)
            prev = self.spk_emb_layer(prev)

        dec_layer_out = prev

        # decoder layers
        if mode == "training":
            for layer in self.glu_layers:
                dec_layer_out = layer(enc_output, dec_layer_out)
        
        elif mode == "inference":
            for layer in self.glu_layers:
                dec_layer_out = layer.incremental_forward(enc_output, dec_layer_out)

        dec_output = self.conv_o(dec_layer_out)
        dec_output = dec_output.permute(0, -1, -2)  # (B, T, C)
        dec_output = dec_output.contiguous().view(B, -1, D)   
        dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)
        return dec_output

    def reset_state(self):
        for layer in self.glu_layers:
            layer.clear_buffer()
