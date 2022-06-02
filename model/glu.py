"""
Gated Linear Unit
"""

import os
os.environ['PYTHONBREAKPOINT'] = ''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .transformer import Prenet


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels*2, kernel_size, padding=self.padding, **kwargs))

    def forward(self, x):
        # 1 次元畳み込み
        y = self.conv(x)
        # 因果性を担保するために、順方向にシフトする
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        return y


class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropput=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropput)
        self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, feature, x):
        """
        feature（encoderからの特徴量） : (B, C, T)
        x（音響特徴量） : (B, C, T)
        """
        res = x
        y = self.dropout(x)
        y = self.causal_conv(y)
        y1, y2 = torch.split(y, y.shape[1] // 2, dim=1)

        feature = F.softsign(self.conv(feature))
        # y2 += feature
        y2 = y2 + feature
        out = F.sigmoid(y1) * y2

        out += res
        out *= 0.5 ** 0.5
        return out


class GLU(nn.Module):
    def __init__(
        self, inner_channels, out_channels,
        pre_in_channels, pre_inner_channels,
        reduction_factor=2, n_layers=4):

        super().__init__()

        self.prenet = Prenet(pre_in_channels, inner_channels, pre_inner_channels)
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        self.glu_stack = nn.ModuleList([
            GLUBlock(inner_channels, inner_channels) for _ in range(n_layers)
        ])

        self.conv_o = weight_norm(nn.Conv1d(
            inner_channels, self.out_channels * self.reduction_factor, kernel_size=1))

    def forward(self, enc_output, prev=None, gc=None):
        """
        reduction_factorにより、
        口唇動画のフレームの、reduction_factor倍のフレームを同時に出力する

        input
        prev : (B, C, T=300)
        enc_output : (B, T=150, C)

        return
        dec_output : (B, C, T=300)
        """
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels
        # 前時刻の出力
        self.pre_out = None

        # global conditionの結合
        
        # reshape for reduction factor
        if prev is not None:
            prev = prev.permute(0, -1, -2)
            prev = prev.reshape(B, -1, D * self.reduction_factor)
            prev = prev.permute(0, -1, -2)  

        # Prenet
        prev = self.prenet(prev)    # (B, d_model, T=150)
        self.pre_out = prev

        # GLU stack
        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)
        dec_output = prev

        for layer in self.glu_stack:
            dec_output = layer(enc_output, dec_output)

        dec_output = self.conv_o(dec_output)
        dec_output = dec_output.reshape(B, D, -1)   
        
        return dec_output   

    def inference(self, enc_output, prev=None, gc=None):
        """
        reduction_factorにより、
        口唇動画のフレームの、reduction_factor倍のフレームを同時に出力する

        input
        enc_output : (B, T, C)

        return
        dec_output : (B, C, T)
        """
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels
        # 前時刻の出力
        self.pre_out = None
        
        # global conditionの結合
        
        # reshape for reduction factor
        if prev is not None:
            prev = prev.permute(0, -1, -2)
            prev = prev.reshape(B, -1, D * self.reduction_factor)
            prev = prev.permute(0, -1, -2)
        else:
            go_frame = torch.zeros(B, D * self.reduction_factor, 1)
            prev = go_frame

        max_decoder_time_steps = T

        # メインループ
        outs = []
        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)
        for _ in range(max_decoder_time_steps):
            # Prenet
            pre_out = self.prenet(prev)    # (B, d_model, T=150)
            
            # GLU stack
            dec_output = pre_out
            breakpoint()
            for layer in self.glu_stack:
                dec_output = layer(enc_output, dec_output)

            dec_output = self.conv_o(dec_output)

            # 出力を保持
            outs.append(dec_output.reshape(B, D, -1)[:, :, -2:])

            # 次のループへの入力
            prev = torch.cat((prev, dec_output[:, :, -1].unsqueeze(-1)), dim=2)
        
        # 各時刻の出力を結合
        outs = torch.cat(outs, dim=2)
        return outs


def main():
    # 3DResnetからの出力
    batch = 8
    channels = 256
    t = 150
    enc_output = torch.rand(batch, channels, t)
    enc_output = enc_output.permute(0, 2, 1)    # (B, T, C)

    # 音響特徴量
    feature_channels = 80
    feature = torch.rand(batch, feature_channels, t*2)   # (B, C, T)

    # GLU
    inner_channels = 256
    out_channels = feature_channels
    pre_in_channels = feature_channels * 2
    pre_inner_channels = 32
    net = GLU(inner_channels, out_channels, pre_in_channels, pre_inner_channels)
    out = net(enc_output, feature)



if __name__ == "__main__":
    main()