"""
reference
https://github.com/Chris10M/Lip2Speech.git
"""

import os
import sys
# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from hparams import create_hparams


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output


class MelSpectrogram(torch.nn.Module):
    def __init__(self, hparams=create_hparams()):        
        super(MelSpectrogram, self).__init__()

        self.mel_spec = T.MelSpectrogram(
                                        sample_rate=hparams.sampling_rate, 
                                        n_fft=hparams.n_fft, 
                                        win_length=hparams.win_length, 
                                        hop_length=hparams.hop_length,
                                        f_min=hparams.f_min,
                                        f_max=hparams.f_max,
                                        n_mels=hparams.n_mel_channels
                                    )


    def forward(self, waveform):
        melspec = self.mel_spec(waveform)
        melspec = spectral_normalize(melspec)

        return melspec