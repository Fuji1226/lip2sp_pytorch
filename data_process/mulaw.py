"""
音声波形を圧縮するmulaw量子化
Wavenetで使用されている
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