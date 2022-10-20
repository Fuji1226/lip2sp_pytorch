import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

from .conv import Conv1d


class UpsampleNetwork(nn.Module):
    def __init__(self, upsample_scales):
        super().__init__()
        self.upsample_scales = upsample_scales
        self.conv_layers = nn.ModuleList()
        for scale in upsample_scales:
            kernel_size = (1, scale * 2 + 1)
            conv = nn.Conv2d(
                1, 1, kernel_size=kernel_size, padding=(0, scale), bias=False
            )
            conv.weight.data.fill_(1.0 / np.prod(kernel_size))
            self.conv_layers.append(conv)

    def forward(self, feature):
        """
        feature : (B, C, T)
        """
        feature = feature.unsqueeze(1)  # (B, 1, C, T)

        # 最近傍補完->畳み込みの繰り返しで，時間方向にアップサンプリング
        for idx, scale in enumerate(self.upsample_scales):
            feature = F.interpolate(feature, scale_factor=(1, scale), mode="nearest")
            feature = self.conv_layers[idx](feature)
        
        return feature.squeeze(1)


class ConvInUpsampleNetwork(nn.Module):
    def __init__(self, upsample_scales, feature_channels, aux_context_window=2):
        super().__init__()
        # 条件付け特徴量近傍を、1 次元畳み込みによって考慮します
        kernel_size = 2 * aux_context_window + 1
        padding = (kernel_size - 1) // 2
        self.conv_in = Conv1d(feature_channels, feature_channels, kernel_size, padding=padding, bias=False)
        self.upsample = UpsampleNetwork(upsample_scales)

    def forward(self, feature):
        return self.upsample(self.conv_in(feature))






import librosa
def main():
    audio_path = "/Users/minami/dataset/lip/cropped/F01_kablab/ATR503_j01_0.wav"
    wav, fs = librosa.load(audio_path, sr=None)
    print(wav.shape)
    print(fs)

    win_length = 640
    hop_length = win_length // 4
    n_mels = 80
    fmin = 0
    fmax = 7600

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=fs,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = torch.from_numpy(mel)
    mel = mel.unsqueeze(0)
    print(mel.shape)

    upsample_scales = [2, 4, 4, 5]
    net = UpsampleNetwork(upsample_scales)
    # print(net)

    mel_out = net(mel[..., :-1])
    print(mel_out.shape)

    net = ConvInUpsampleNetwork(upsample_scales, 80, 2)
    mel_out = net(mel[..., :-1])
    print(mel_out.shape)

if __name__ == "__main__":
    main()