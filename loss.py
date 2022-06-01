"""
損失関数
"""

import torch
from model.transformer import make_pad_mask


def masked_mse(output, target, data_len, max_len=300-2):
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
    breakpoint()
    mse /= idx
    return mse


def main():
    x = torch.rand(8, 1, 10)
    y = torch.rand(8, 1, 10)
    loss = (x - y)**2
    loss_a = torch.zeros(1, loss.shape[1], 1)
    breakpoint()
    loss_a += loss[0, :, 0]
    breakpoint()
        


if __name__ == "__main__":
    main()