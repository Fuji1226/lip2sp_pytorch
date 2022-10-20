"""
損失関数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_remake import make_pad_mask_for_loss
from data_process.feature import delta_feature, blur_pooling2D


class MaskedLoss:
    def __init__(self):
        pass
        
    def mse_loss(self, output, target, data_len, max_len):
        """
        パディングされた部分を考慮し、損失計算から省いたMSE loss
        output, target : (B, C, T)
        """
        # マスク作成
        mask = make_pad_mask_for_loss(data_len, max_len, output) 

        # 二乗誤差を計算
        loss = (output - target)**2

        # maskがFalseのところは0にして平均を取る
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, T)

        # maskしていないところ全体で平均
        mask = mask.squeeze(1)  # (B, T)
        n_loss = torch.where(mask == 0, torch.ones_like(mask).to(torch.float32), torch.zeros_like(mask).to(torch.float32))
        mse_loss = torch.sum(loss) / torch.sum(n_loss)

        return mse_loss

    def delta_loss(self, output, target, data_len, max_len, device, blur, batch_norm):
        """
        音響特徴量の動的特徴量についての損失関数
        """
        # 田口さんのやつに変更
        B, C, T = output.shape
        if blur:
            output = blur_pooling2D(output, device)
            target = blur_pooling2D(target, device)

        # 動的特徴量の計算  (B, C, T) -> (B, 3 * C, T)
        output = delta_feature(output) 
        target = delta_feature(target)

        if batch_norm:
            bn = nn.BatchNorm2d(3, affine=False).to(device)
            output = bn(output.reshape(B, 3, -1, T))
            target = bn(target.reshape(B, 3, -1, T))
            output = output.reshape(B, -1, T)
            target = target.reshape(B, -1, T)
        
        # 各チャンネルごとの標準偏差（ブロードキャストのため次元を増やしてます）
        target_std = torch.std(target, dim=(0, -1)).unsqueeze(0).unsqueeze(-1)

        mask = make_pad_mask(data_len, max_len)

        if batch_norm:
            loss = (output - target) ** 2
        else:
            loss = ((output - target) / target_std)**2

        loss = torch.where(mask == 0, loss, torch.tensor(0).to(device=loss.device, dtype=loss.dtype))
        loss = torch.mean(loss, dim=1)
        ones = torch.ones_like(mask).to(device=loss.device, dtype=loss.dtype)
        n_loss = torch.where(mask == 0, ones, torch.tensor(0).to(device=loss.device, dtype=loss.dtype))
        mse_loss = torch.sum(loss) / torch.sum(n_loss)

        return mse_loss


class AdversarialLoss:
    def __init__(self):
        pass

    def ls_loss(self, out_f, out_r, which_loss):
        # discriminator
        if which_loss == "d":
            # gcあり
            if len(out_f) == 2:
                loss = (out_f[0]**2 + out_f[1]**2) / 2 + ((1 - out_r[0])**2 + (1 - out_r[1])**2) / 2
            # gcなし
            else:
                loss = out_f[0]**2 + (1 - out_r[0])**2

        # generator
        elif which_loss == "g":
            # gcあり
            if len(out_f) == 2:
                loss = ((1 - out_f[0])**2 + (1 - out_f[1])**2) / 2
            # gcなし
            else:
                loss = (1 - out_f[0])**2
        
        loss = torch.mean(loss, dim=0)
        return loss

    def fm_loss(self, fmaps_f, fmaps_r, data_len=None, max_len=None):
        """
        feature matching loss
        generatorに使う
        """
        # # data_lenとmax_lenを出力の長さを考慮して調節
        # data_lens, max_lens = [], []
        # for fmap in fmaps_f:
        #     ratio = fmap.shape[-1] / max_len
        #     data_len = data_len * ratio
        #     data_lens.append(data_len.to(torch.int64))
        #     max_lens.append(int(max_len * ratio))

        # # mask
        # masks = []
        # for data_len, max_len in zip(data_lens, max_lens):
        #     masks.append(make_pad_mask(data_len, max_len))
        
        # # loss
        # losses = []
        # for fmap_f, fmap_r, mask in zip(fmaps_f, fmaps_r, masks):
        #     loss = torch.abs(fmap_f - fmap_r)
        #     loss = torch.where(mask == 1, loss, torch.tensor(0).to(device=loss.device, dtype=loss.dtype))
        #     loss = torch.mean(loss, dim=1)
        #     loss = torch.sum(loss) / torch.sum(mask)
        #     losses.append(loss)

        losses = []
        for fmap_f, fmap_r in zip(fmaps_f, fmaps_r):
            loss = F.l1_loss(fmap_f, fmap_r)
            losses.append(loss)
        
        loss = sum(losses) / len(losses)
        return loss


def main():
    data_len = torch.tensor([300, 400, 200])
    x = torch.rand(3, 80, 300)
    y = torch.rand(3, 80, 300)
    loss_f = MaskedLoss()
    loss = loss_f.mse_loss(x, y, data_len, x.shape[-1])
    print(loss)


if __name__ == "__main__":
    main()