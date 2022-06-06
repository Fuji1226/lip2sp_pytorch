"""
損失関数
"""


import torch
from model.transformer import make_pad_mask
from model.discriminator import UNetDiscriminator, JCUDiscriminator
from data_process.feature import delta_feature


def masked_mse(output, target, data_len, max_len):
    """
    パディングされた部分を考慮し、損失計算から省いたMSE loss
    output, target : (B, C, T)
    """
    # マスク作成
    mask = make_pad_mask(data_len, max_len)    

    # 二乗誤差を計算
    loss = (output - target)**2

    # マスクでパディング部分を隠す
    mse = 0
    idx = 0
    for i in range(loss.shape[0]):
        for j in range(loss.shape[-1]):
            if mask[i, :, j]:
                mse += torch.mean(loss[i, :, j])
                idx += 1
            else:
                break
    # 平均
    mse /= idx
    return mse


def delta_loss(output, target, data_len, max_len):
    """
    音響特徴量の動的特徴量についての損失関数
    """
    # マスク
    mask = make_pad_mask(data_len, max_len)

    # 動的特徴量の計算  (B, C, T) -> (B, 3 * C, T)
    output = delta_feature(output) 
    target = delta_feature(target)

    # 各チャンネルごとの標準偏差（ブロードキャストのため次元を増やしてます）
    target_std = torch.std(target, dim=(0, -1)).unsqueeze(0).unsqueeze(-1)

    # 正規化されたメルスペクトログラムの静的・動的特徴量の二乗誤差
    loss = ((output - target) / target_std)**2
    
    # マスクでパディング部分を隠す
    mse = 0
    idx = 0
    for i in range(loss.shape[0]):
        for j in range(loss.shape[-1]):
            if mask[i, :, j]:
                mse += torch.mean(loss[i, :, j])
                idx += 1
            else:
                break
    # 平均
    mse /= idx
    return mse


def ls_loss(out_f, out_r, data_len, max_len, which_d, which_loss):
    """
    least squares loss 
    discriminator、generator共に用いる
    
    which_dは使用したdiscriminator
    which_lossでdiscriminatorとgeneratorを切り替え
    """
    assert which_d == "unet" or "jcu", "please set which_d unet or jcu"
    assert which_loss == "d" or "g", "please set which_loss d or g"

    # encoderとdecoderの出力の形状が異なるため、マスクするためのdata_lenをmax_lenとの比率から計算
    ratios = []
    for layer in range(len(out_f)):
        ratios.append(out_f[layer].shape[-1] / max_len)
    
    # 各層ごとの比率に合わせて、data_lenをスケーリング
    max_lens = []
    data_lens = []
    for layer in range(len(out_f)):
        max_lens.append(int(ratios[layer] * max_len))
        data_lens.append(ratios[layer] * data_len)

    # get mask
    masks = []
    for layer in range(len(out_f)):
        masks.append(make_pad_mask(data_lens[layer], max_lens[layer]))
    
    # discriminatorに対する損失を計算する場合
    losses = []
    if which_loss == "d":
        # JCU discriminatorかつ、global conditionがある場合
        if which_d == "jcu" and len(out_f) == 2:
            losses.append(
                (out_f[0]**2 + out_f[1]**2) / 2 + ((out_r[0] - 1)**2 + (out_r[1] - 1)**2) / 2
            )
        # global conditionがない場合
        else:
            for layer in range(len(out_f)):
                # 実データを1、generatorが生成したデータを0にするように学習（見分けたい）
                losses.append((1 - out_r[layer])**2 + out_f[layer]**2)
                if which_d == "unet":
                    losses[layer] = losses[layer].squeeze(1)    # (B, 1, C, T) -> (B, C, T)

    # generatorに対する損失を計算する場合
    else:
        # JCU discriminatorかつ、global conditionがある場合
        if which_d == "jcu" and len(out_f) == 2:
            losses.append(
                ((out_f[0] - 1)**2 + (out_f[1] - 1)**2) / 2
            )
        # global conditionがない場合
        else:
            for layer in range(len(out_f)):
                # generatorが生成したデータを1にするように学習（騙したい）
                losses.append((1 - out_f[layer])**2)
                if which_d == "unet":
                    losses[layer] = losses[layer].squeeze(1)    # (B, 1, C, T) -> (B, C, T)

    # 損失計算
    losses_mean = []
    for layer in range(len(losses)):
        # 層ごとに初期化
        calc_mean = 0
        calc_idx = 0
        for i in range(losses[layer].shape[0]):
            for j in range(losses[layer].shape[-1]):
                if masks[layer][i, :, j]:
                    # 各層、各バッチ、各時刻ごとの平均値
                    calc_mean += torch.mean(losses[layer][i, :, j])
                    calc_idx += 1
                else:
                    break
        # 各層における平均値
        losses_mean.append(calc_mean / calc_idx)

    return sum(losses_mean)


def fm_loss(fmaps_f, fmaps_r, data_len, max_len, which_d):
    """
    feature matching loss
    generatorに対してのみ用いる
    discriminatorの各層における特徴量についての損失

    which_dは使用したdiscriminator
    """
    assert which_d == "unet" or "jcu", "please set which_d unet or jcu"
    # encoder、decoder各層におけるmax_lenに対しての比率を計算
    ratios = []
    for layer in range(len(fmaps_f)):
        ratios.append(fmaps_f[layer].shape[-1] / max_len)

    # 各層ごとの比率に合わせて、data_lenをスケーリング
    max_lens = []
    data_lens = []
    for layer in range(len(fmaps_f)):
        max_lens.append(int(ratios[layer] * max_len))
        data_lens.append(ratios[layer] * data_len)

    # 各層の比率にあったマスク行列を生成
    masks = []
    for layer in range(len(fmaps_f)):
        masks.append(make_pad_mask(data_lens[layer], max_lens[layer]))

    # discriminatorの各層における特徴量の誤差を計算
    losses = []
    for layer in range(len(fmaps_f)):
        losses.append(torch.abs(fmaps_f[layer] - fmaps_r[layer]))

    # 各層ごとにマスクされる部分以外から平均を計算
    losses_mean = []
    for layer in range(len(losses)):
        # 層ごとに初期化
        calc_mean = 0
        calc_idx = 0
        for i in range(losses[layer].shape[0]):
            for j in range(losses[layer].shape[-1]):
                if masks[layer][i, :, j]:
                    # 各層、各バッチ、各時刻ごとの平均値
                    if which_d == "unet":
                        calc_mean += torch.mean(losses[layer][i, :, :, j])
                    elif which_d == "jcu":
                        calc_mean += torch.mean(losses[layer][i, :, j])
                    calc_idx += 1
                else:
                    break
        # 各層における平均値
        losses_mean.append(calc_mean / calc_idx)
    
    return sum(losses_mean) / len(fmaps_f)  # 全層の平均値


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

    loss_mask = masked_mse(output, target, data_len, max_len)
    print(f"masked_mse_loss = {loss_mask}")
    loss_delta = delta_loss(output, target, data_len, max_len)
    print(f"delta_loss = {loss_delta}")

    gc = torch.arange(B).reshape(B, 1)      # 複数話者の場合を想定
    use_gc = True

    # ls_lossの切り替え
    which_loss = "d"

    print("JCU")
    # JCU discriminator
    d_J = JCUDiscriminator(mel_channels, 1, use_gc, emb_in=B)
    if use_gc:
        out_f, fmaps_f = d_J(output, gc)
        out_r, fmaps_r = d_J(target, gc)
    else:
        out_f, fmaps_f = d_J(output, )
        out_r, fmaps_r = d_J(target, )
   
    # ls_loss
    ls_jcu = ls_loss(out_f, out_r, data_len, max_len, which_d="jcu", which_loss=which_loss)
    print(ls_jcu)
    
    # fm_loss
    fm_JCU = fm_loss(fmaps_f, fmaps_r, data_len, max_len, which_d="jcu")
    print(fm_JCU)

    # print("Unet")
    # # U_net discriminator
    # d_U = UNetDiscriminator()
    # out_f, fmaps_f = d_U(output)
    # out_r, fmaps_r = d_U(target)

    # # ls_loss
    # ls_u = ls_loss(out_f, out_r, data_len, max_len, which_d="unet", which_loss=which_loss)
    # print(ls_u)

    # # fm_loss
    # fm_unet = fm_loss(fmaps_f, fmaps_r, data_len, max_len, which_d="unet")
    # print(fm_unet)


if __name__ == "__main__":
    main()