"""
glu
transformer_remake.pyと一緒に使えるようになっています
transformer_taguchi.pyとは使えません(データ形状が違うので)
"""
import sys
from pathlib import Path
sys.path.append(Path("~/lip2sp_pytorch").expanduser())

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .transformer_remake import shift
    from .pre_post import ProgressivePrenet
    from wavenet.model.conv import CausalConv1d   
except:
    from transformer_remake import shift
    from pre_post import ProgressivePrenet
    from wavenet.model.conv import CausalConv1d   


class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropput=0.5):
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

        # enc_output = F.softsign(self.conv(enc_output)).clone()
        enc_output = F.softsign(self.conv(enc_output))
        y2 = y2 + enc_output
        out = torch.sigmoid(y1) * y2

        out += res
        out *= 0.5 ** 0.5
        return out

    def clear_buffer(self):
        self.causal_conv.clear_buffer()


class ProgressiveGLU(nn.Module):
    def __init__(
        self, inner_channels, out_channels, pre_in_channels, pre_inner_channels,
        reduction_factor=2, n_layers=4, kernel_size=5, dropout=0.5):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        self.prenet = ProgressivePrenet(
            in_channels=pre_in_channels, 
            out_channels=inner_channels, 
            reduction_factor=reduction_factor,
            inner_channels=pre_inner_channels,
        )
        self.dropout = nn.Dropout(dropout)
        self.glu_layers = nn.ModuleList([
            GLUBlock(inner_channels, inner_channels, kernel_size, dropout) for _ in range(n_layers)
        ])

        self.last_conv40 = nn.Conv1d(inner_channels, 40 * reduction_factor, kernel_size=1)
        self.last_conv60 = nn.Conv1d(inner_channels, 60 * reduction_factor, kernel_size=1)
        self.last_conv80 = nn.Conv1d(inner_channels, 80 * reduction_factor, kernel_size=1)

    def forward(self, enc_output, target=None, gc=None, mode=None):
        """
        enc_output : (B, T, C)
        target : (B, C, T)
        """
        assert mode == "training" or "inference"
        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)
        B = enc_output.shape[0]
        T = enc_output.shape[-1]
        D = self.out_channels

        # target shift
        if mode == "training":
            target = shift(target, self.reduction_factor)

        # view for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)  # (B, T, C)
            target = target.contiguous().view(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  # (B, C, T)
        else:
            target = torch.zeros(B, D * self.reduction_factor, 1).to(device=enc_output.device, dtype=enc_output.dtype) 

        # prenet
        target = self.dropout(self.prenet(target))
        # dec_layer_out = target.clone()
        dec_layer_out = target

        # decoder layers
        if mode == "training":
            for layer in self.glu_layers:
                dec_layer_out = layer(enc_output, dec_layer_out)
        
        elif mode == "inference":
            for layer in self.glu_layers:
                dec_layer_out = layer.incremental_forward(enc_output, dec_layer_out)

        # convolution
        if self.out_channels == 40:
            dec_output = self.last_conv40(dec_layer_out)
        elif self.out_channels == 60:
            dec_output = self.last_conv60(dec_layer_out)
        elif self.out_channels == 80:
            dec_output = self.last_conv80(dec_layer_out)

        dec_output = dec_output.permute(0, -1, -2)  # (B, T, C)
        dec_output = dec_output.contiguous().view(B, -1, D)   
        dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)
        return dec_output

    def reset_state(self):
        for layer in self.glu_layers:
            layer.clear_buffer()