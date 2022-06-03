"""
損失関数
"""

import torch
from model.transformer import make_pad_mask


def masked_mse(output, target, data_len, max_len):
    """
    パディングされた部分を考慮し、損失計算から省く
    output, target : (B, C, T)
    """
    
    loss = (output - target)**2
    mask = make_pad_mask(data_len, max_len)
    mse = 0
    idx = 0
    for i in range(loss.shape[0]):
        for j in range(loss.shape[-1]):
            if mask[i, :, j]:
                mse += torch.mean(loss[i, :, j])
                idx += 1
            else:
                break
    
    mse /= idx
    return mse
