"""
user/minami/dataset/lip/lip_cropped
このディレクトリに口唇部分を切り取った動画と、wavデータを入れておけば動くと思います!

datasetを変更しました
"""
from omegaconf import DictConfig, OmegaConf
import hydra

# import wandb
# wandb.init(
#     project='llip2sp_pytorch',
#     name="desk-test"
# )

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_no_chainer import KablabDataset, KablabTransform
from model.models import Lip2SP
from loss import masked_mse, delta_loss, ls_loss, fm_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator


current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# パラメータの保存
def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def make_train_loader(cfg):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    datasets = KablabDataset(
        data_root=cfg.model.train_path,
        train=True,
        transforms=trans,
        cfg=cfg,
    )
    train_loader = DataLoader(
        dataset=datasets,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )
    return train_loader


def make_test_loader(cfg):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    datasets = KablabDataset(
        data_root=cfg.model.test_path,
        train=False,
        transforms=trans,
        cfg=cfg,
    )
    test_loader = DataLoader(
        dataset=datasets,
        # batch_size=cfg.train.batch_size,
        batch_size=1,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader
    

def train_one_epoch(model: nn.Module, discriminator, data_loader, optimizer, loss_f, device):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(data_loader)
    print("iter start")
    for batch in data_loader:
        model.train()
        model = model.to(device)
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        (lip, target, feat_add), data_len = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        ################順伝搬###############
        # output : postnet後の出力
        # dec_output : postnet前の出力
        output, dec_output = model(
            lip=lip,
            data_len=data_len,
            prev=target,
        )                      
        ####################################
        
        loss_default = loss_f(output[:, :, :-2], target[:, :, 2:]) + loss_f(dec_output[:, :, :-2], target[:, :, 2:])
        # パディング部分をマスクした損失を計算。postnet前後の出力両方を考慮。
        loss = masked_mse(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + masked_mse(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # wandb.log({"train_iter_loss": loss.item()})

    epoch_loss /= data_cnt
    return epoch_loss


def train_one_epoch_with_d(model: nn.Module, discriminator, data_loader, optimizer, optimizer_d, loss_f, device, cfg):
    assert cfg.model.which_d is not None, "discriminatorが設定されていません!"

    epoch_loss = 0
    data_cnt = 0
    for batch in data_loader:
        model.train()
        discriminator.train()
        
        (lip, target, feat_add), data_len = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size

        #====================================================
        # discriminatorの最適化
        # generator1回に対して複数回最適化するコードもあり。とりあえず1回で実装。
        #====================================================
        # discriminatorのパラメータ更新を行うように設定
        for param in discriminator.parameters():
            param.requires_grad = True

        ################順伝搬###############
        # output : postnet後の出力
        # dec_output : postnet前の出力
        # generatorのパラメータが更新されないように設定
        with torch.no_grad():
            output, dec_output = model(
                lip=lip,
                data_len=data_len,
                prev=target,
            )                      
        ####################################
        
        # discriminatorへ入力
        try:
            out_f, fmaps_f = discriminator(output[:, :, :-2])   # 生成データを入力
            out_r, fmaps_r = discriminator(target[:, :, 2:])    # 実データを入力
            # 損失計算
            loss_d = ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="d")
        except Exception:
            print("error")
        
        # 最適化
        loss_d.backward()
        optimizer_d.step()

        #====================================================
        # generatorの最適化
        #====================================================
        # discriminatorのパラメータ更新を行わないように設定
        for param in discriminator.parameters():
            param.requires_grad = False
        
        # generatorでデータ生成
        output, dec_output = model(
            lip=lip,
            data_len=data_len,
            prev=target,
        )

        # discriminatorへ入力
        try:
            out_f, fmaps_f = discriminator(output[:, :, :-2])   # 生成データを入力
            out_r, fmaps_r = discriminator(target[:, :, 2:])    # 実データを入力
            # 損失計算
            loss_g_ls = ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="g")
            loss_g_fm = fm_loss(fmaps_f, fmaps_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d)
        except Exception:
            print("error")

        loss_default = loss_f(output[:, :, :-2], target[:, :, 2:]) + loss_f(dec_output[:, :, :-2], target[:, :, 2:])
        loss_mask = masked_mse(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + masked_mse(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 
        loss_delta = delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 

        # postnet前後のmse_lossと、GAN関連のlossの和（実際は重みづけが必要そう）
        loss = masked_mse(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + masked_mse(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                + delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                    + loss_g_ls + loss_g_fm
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


    epoch_loss /= data_cnt
    return epoch_loss


def save_result(loss_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(save_path)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    ###ここにデータセットモデルのインスタンス作成train関数を回す#####

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    #resultの表示
    result_path = 'results'
    os.makedirs(result_path, exist_ok=True)

    # モデルのパラメータの保存
    model_save_path = cfg.model.save_path
    os.makedirs(model_save_path, exist_ok=True)

    #インスタンス作成
    model = Lip2SP(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        glu_inner_channels=cfg.model.glu_inner_channels,
        glu_layers=cfg.model.glu_layers,
        pre_in_channels=cfg.model.pre_in_channels,
        pre_inner_channels=cfg.model.pre_inner_channels,
        post_inner_channels=cfg.model.post_inner_channels,
        n_position=cfg.model.length * 5,
        max_len=cfg.model.length // 2,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        training_method=cfg.train.training_method,
        num_passes=cfg.train.num_passes,
        mixing_prob=cfg.train.mixing_prob,
        dropout=cfg.train.dropout,
        reduction_factor=cfg.model.reduction_factor,
        use_gc=cfg.train.use_gc
    )
    
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, betas=(cfg.train.beta_1, cfg.train.beta_2)
    )

    # discriminator
    if cfg.model.which_d is not None:
        if cfg.model.which_d == "unet":
            discriminator = UNetDiscriminator()
        elif cfg.model.which_d == "jcu":
            discriminator = JCUDiscriminator(
                in_channels=cfg.model.out_channels,
                out_channels=1,
                use_gc=cfg.train.use_gc,
                emb_in=cfg.train.batch_size,
            )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=cfg.train.lr, betas=(cfg.train.beta_1, cfg.train.beta_2)
        )
    else:
        discriminator = None

    # Dataloader作成
    train_loader = make_train_loader(cfg)
    test_loader = make_test_loader(cfg)
    ##########################################################################################
    # 損失関数
    loss_f = nn.MSELoss()
    train_loss_list = []

    # training
    if cfg.model.which_d is None:
        for epoch in range(cfg.train.max_epoch):
            print(f"##### {epoch} #####")
            epoch_loss = train_one_epoch(model, discriminator, train_loader, optimizer, loss_f, device)
            train_loss_list.append(epoch_loss)
            print(f"epoch_loss = {epoch_loss}")
            print(f"train_loss_list = {train_loss_list}")

        save_result(train_loss_list, result_path+'/train_loss.png')
        torch.save(model.state_dict(), model_save_path+'/model_world.pth')
        
    else:
        for epoch in range(cfg.train.max_epoch):
            print(f"##### {epoch} #####")
            epoch_loss = train_one_epoch_with_d(model, discriminator, train_loader, optimizer, optimizer_d, loss_f, device, cfg)
            train_loss_list.append(epoch_loss)
            print(f"epoch_loss = {epoch_loss}")
            print(f"train_loss_list = {train_loss_list}")
        
        save_result(train_loss_list, result_path+'/train_loss.png')
        torch.save(model.state_dict(), model_save_path+'model.pth')

if __name__=='__main__':
    main()