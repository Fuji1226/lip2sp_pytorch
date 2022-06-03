"""
損失関数
"""

import torch
from model.transformer import make_pad_mask
from model.discriminator import UNetDiscriminator


def masked_mse(output, target, data_len, max_len):
    """
    パディングされた部分を考慮し、損失計算から省く
    output, target : (B, C, T)
    """

    mask = make_pad_mask(data_len, max_len)    
    loss = (output - target)**2
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


def ls_loss_Unet(out_enc_f, out_dec_f, out_enc_r, out_dec_r, data_len, max_len, which_loss):
    """
    least squares loss for Unet-discriminator
    """
    assert which_loss == "d" or which_loss == "g", "please set which_loss"

    # encoderとdecoderの出力の形状が異なるため、マスクするためのdata_lenをmax_lenとの比率から計算
    ratio = out_enc_f.shape[-1] / out_dec_f.shape[-1]   # スケールの比率
    max_len_enc = int(max_len * ratio)
    data_len_enc = data_len * ratio
    max_len_dec = max_len
    data_len_dec = data_len

    # get mask
    mask_enc = make_pad_mask(data_len_enc, max_len_enc)
    mask_dec = make_pad_mask(data_len_dec, max_len_dec)

    if which_loss == "d":
        # loss calculation
        loss_enc = (1 - out_enc_r)**2 + out_enc_f**2   # (B, 1, C//8, T//8)
        loss_dec = (1 - out_dec_r)**2 + out_dec_f**2   # (B, 1, C, T)
        loss_enc = loss_enc.squeeze(1)  # (B, C//8, T)
        loss_dec = loss_dec.squeeze(1)  # (B, C, T)

        loss_enc_mean = 0
        loss_dec_mean = 0
        idx_enc = 0
        idx_dec = 0
        for i in range(loss_enc.shape[0]):
            for j in range(loss_enc.shape[-1]):
                if mask_enc[i, :, j]:
                    loss_enc_mean += torch.mean(loss_enc[i, :, j])
                    idx_enc += 1
                else:
                    break
        for i in range(loss_dec.shape[0]):
            for j in range(loss_dec.shape[-1]):
                if mask_dec[i, :, j]:
                    loss_dec_mean += torch.mean(loss_dec[i, :, j])
                    idx_dec += 1
                else:
                    break
        loss_enc_mean /= idx_enc
        loss_dec_mean /= idx_dec
    
    # else:

    
    return


def fm_loss_Unet(output, target, data_len, max_len):
    """
    feature matching loss for Unet-discriminator
    """
    return


def main():
    # 系列長
    data_len = [500, 500, 400, 400, 300, 300, 200, 200]
    data_len = torch.tensor(data_len)
    max_len = 400

    B = 8
    mel_channels = 80
    t =400
    output = torch.rand(B, mel_channels, t)
    target = torch.rand(B, mel_channels, t)

    discriminator = UNetDiscriminator()
    out_enc_f, fmaps_enc_f, out_dec_f, fmaps_dec_f = discriminator(output)
    out_enc_r, fmaps_enc_r, out_dec_r, fmaps_dec_r = discriminator(target)

    # ls_loss
    ls_loss_Unet(out_enc_f, out_dec_f, out_enc_r, out_dec_r, data_len, max_len, which_loss="d")



if __name__ == "__main__":
    main()