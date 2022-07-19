"""
16ビットの音声を圧縮するためのmulawアルゴリズム
"""

import numpy as np


def mulaw(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def quantize(y, mu=255, offset=1):
    return ((y + offset) / 2 * mu).astype(np.int64)


def mulaw_quantize(x, mu=255):
    return quantize(mulaw(x, mu), mu)


def inv_mulaw(y, mu=255):
    return np.sign(y) * (1.0 / mu) * ((1.0 + mu)**np.abs(y) - 1.0)


def inv_quantize(y, mu=255):
    return 2 * y.astype(np.float32) / mu - 1


def inv_mulaw_quantize(y, mu=255):
    return inv_mulaw(inv_quantize(y, mu), mu)




import librosa
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.io.wavfile import write
import torch
import torch.nn.functional as F
def main():
    audio_path = "/Users/minami/dataset/lip/cropped/F01_kablab/ATR503_j01_0.wav"
    wav, fs = librosa.load(audio_path, sr=None)
    print(wav.shape)
    print(fs)

    wav_mulaw = mulaw(wav)
    wav_mulaw_quantize = mulaw_quantize(wav)
    wav_recon = inv_mulaw_quantize(wav_mulaw_quantize)

    save_path = "/Users/minami/dataset/mulaw"
    os.makedirs(save_path, exist_ok=True)

    write(
        filename=os.path.join(save_path, f"{Path(audio_path).stem}_mulaw.wav"),
        rate=fs,
        data=wav_mulaw,
    )
    write(
        filename=os.path.join(save_path, f"{Path(audio_path).stem}_recon.wav"),
        rate=fs,
        data=wav_recon,
    )

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(wav)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(wav_mulaw)
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(wav_mulaw_quantize)
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(wav_recon)
    fig.savefig(os.path.join(save_path, "waveform.png"))

    print(wav_mulaw_quantize.shape)
    print(wav_mulaw_quantize)
    wav_mulaw_quantize = torch.from_numpy(wav_mulaw_quantize)
    wav_mulaw_quantize = F.one_hot(wav_mulaw_quantize, 256)
    print(wav_mulaw_quantize.shape)
    print(wav_mulaw_quantize)
    

if __name__ == "__main__":
    main()
