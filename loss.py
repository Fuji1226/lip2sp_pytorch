"""
損失関数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_remake import make_pad_mask


class MaskedLoss:
    def __init__(self, weight=None, use_weighted_mean=False):
        self.weight = weight
        self.use_weighted_mean = use_weighted_mean
        
    def mse_loss(self, output, target, data_len, max_len):
        """
        パディングされた部分を考慮し、損失計算から省いたMSE loss
        output, target : (B, C, T) or (B, C, H, W, T)
        """
        # マスク作成
        mask = make_pad_mask(data_len, max_len) 

        # 二乗誤差を計算
        loss = (output - target)**2

        # 動画にも対応するため
        if loss.dim() == 5:
            loss = torch.mean(loss, dim=(2, 3))

        # maskがTrueのところは0にして平均を取る
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, T)

        # maskしていないところ全体で平均
        mask = mask.squeeze(1)  # (B, T)
        n_loss = torch.where(mask == 0, torch.ones_like(mask).to(torch.float32), torch.zeros_like(mask).to(torch.float32))
        mse_loss = torch.sum(loss) / torch.sum(n_loss)

        return mse_loss

    def l1_loss(self, output, target, data_len, max_len):
        mask = make_pad_mask(data_len, max_len)

        loss = torch.abs((output - target))

        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, T)

        mask = mask.squeeze(1)  # (B, T)
        n_loss = torch.where(mask == 0, torch.ones_like(mask).to(torch.float32), torch.zeros_like(mask).to(torch.float32))
        loss = torch.sum(loss) / torch.sum(n_loss)
        return loss

    def cross_entropy_loss(self, output, target, ignore_index, speaker=None):
        if self.use_weighted_mean:
            weight_list = []
            for spk in speaker:
                weight_list.append(self.weight[spk])

            weight = torch.tensor(weight_list).to(device=output.device)

            loss_list = []
            for i in range(output.shape[0]):
                loss_list.append(
                    F.cross_entropy(output[i].unsqueeze(0), target[0].unsqueeze(0), ignore_index=ignore_index) * weight[i]
                )
            loss = sum(loss_list) / len(loss_list)
        else:
            loss = F.cross_entropy(output, target, ignore_index=ignore_index)

        return loss


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
        losses = []
        for fmap_f, fmap_r in zip(fmaps_f, fmaps_r):
            loss = F.l1_loss(fmap_f, fmap_r)
            losses.append(loss)
        
        loss = sum(losses) / len(losses)
        return loss


torch.manual_seed(777)
torch.cuda.manual_seed_all(777)


if __name__ == "__main__":
    net = nn.Conv1d(2, 2, kernel_size=1)
    print("\n### test ###")
    print(net.state_dict())
    optimizer = torch.optim.Adam(
        params=net.parameters(),
        lr=0.1, 
        betas=(0.9, 0.999),
        weight_decay=1e-6,    
    )
    optimizer.zero_grad()

    data_len = torch.tensor([10])
    max_len = 30

    output = torch.rand(1, 2, 30)
    output = net(output)
    # target = torch.rand_like(output) * 10
    target = output.clone().detach()
    target[..., -20:] = 100
    mask = make_pad_mask(data_len, max_len) 

    # print(output)
    # print(target)
    # print(mask)

    mode = 3

    if mode == 1:
        # 二乗誤差を計算
        loss = (output - target)**2
        mse_loss = torch.mean(loss)
    if mode == 2:
        loss = (output - target)**2
        # 動画にも対応するため
        if loss.dim() == 5:
            loss = torch.mean(loss, dim=(2, 3))

        # maskがTrueのところは0にして平均を取る
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, T)

        # maskしていないところ全体で平均
        mask = mask.squeeze(1)  # (B, T)
        n_loss = torch.where(mask == 0, torch.ones_like(mask).to(torch.float32), torch.zeros_like(mask).to(torch.float32))
        mse_loss = torch.sum(loss) / torch.sum(n_loss)
    if mode == 3:
        mask = (1 - mask.to(torch.int)).to(torch.bool)
        output = torch.masked_select(output, mask)
        target = torch.masked_select(target, mask)
        loss = (output - target) ** 2
        mse_loss = torch.mean(loss)

    print(f"mse loss = {mse_loss}")
    mse_loss.backward()
    optimizer.step()
    print(net.state_dict())