import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from conv_block import ResSkipBlock
from upsample import ConvInUpsampleNetwork


class WaveNet(nn.Module):
    def __init__(
        self, out_channels, layers, stacks, residual_channels, gate_channels,
        skip_out_channels, kernel_size, feature_channels, upsample_scales, aux_context_window,
    ):
        super().__init__()
        self.out_channels = out_channels    
        self.feature_channels = feature_channels
        self.aux_context_window = aux_context_window
        self.upsample_scales = upsample_scales

        self.first_conv = weight_norm(nn.Conv1d(out_channels, residual_channels, kernel_size=1))

        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            self.main_conv_layers.append(
                ResSkipBlock(
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    kernel_size=kernel_size,
                    skip_out_channels=skip_out_channels,
                    dilation=dilation,
                    feature_channels=feature_channels,
                )
            )
        
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(),
            weight_norm(nn.Conv1d(skip_out_channels, skip_out_channels, kernel_size=1)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(skip_out_channels, out_channels, kernel_size=1)),
        ])

        self.upsample_net = ConvInUpsampleNetwork(
            upsample_scales=upsample_scales,
            feature_channels=feature_channels,
            aux_context_window=aux_context_window,
        )

    def forward(self, x, feature):
        """
        x : (B, T)
        feature : (B, C, T)
        """
        # 量子化された離散値列から One-hot ベクトルに変換
        # (B, T) -> (B, T, out_channels) -> (B, out_channels, T)
        x = F.one_hot(x, self.out_channels).transpose(1, 2).float()

        # 条件付け特徴量のアップサンプリング
        feature = self.upsample_net(feature)
        assert feature.shape[-1] == x.shape[-1]

        out = self.first_conv(x)

        skips = 0
        for layer in self.main_conv_layers:
            out, skip = layer(out, feature)
            skips += skip

        # スキップ接続の和のみを最終層に入力
        out = skips
        for layer in self.last_conv_layers:
            out = layer(out)
            
        return out


import os
import sys

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(os.path.dirname(__file__))

import librosa
import numpy as np
from data_process.mulaw import mulaw_quantize

def main():
    audio_path = "/Users/minami/dataset/lip/cropped/F01_kablab/ATR503_j01_0.wav"
    x, fs = librosa.load(audio_path, sr=None)
    print(x.shape)
    print(fs)

    win_length = 640
    hop_length = win_length // 4
    n_mels = 80
    fmin = 0
    fmax = 7600

    feature = librosa.feature.melspectrogram(
        y=x,
        sr=fs,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    feature = librosa.power_to_db(feature, ref=np.max)

    x = mulaw_quantize(x)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    feature = torch.from_numpy(feature)
    feature = feature[..., :-1]
    feature = feature.unsqueeze(0)
    
    print(f"x = {x.shape}")
    print(f"feature = {feature.shape}")
    vocoder = WaveNet(
        out_channels=256,
        layers=30,
        stacks=3,
        residual_channels=64,
        gate_channels=128,
        skip_out_channels=64,
        kernel_size=3,
        feature_channels=n_mels,
        upsample_scales=[2, 4, 4, 5],
        aux_context_window=2,
    )
    out = vocoder(
        x=x,
        feature=feature,
    )
    print(f"out = {out.shape}")


if __name__ == "__main__":
    main()