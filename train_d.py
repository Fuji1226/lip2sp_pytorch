"""
discriminatorを使うtrainを分けました
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow
import wandb

from pathlib import Path
from tqdm import tqdm
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_no_chainer import KablabDataset, KablabTransform
from model.models import Lip2SP
from loss import masked_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator
from mf_writer import MlflowWriter
from data_process.feature import delta_feature

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# パラメータの保存
def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def make_train_val_loader(cfg):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    dataset = KablabDataset(
        data_root=cfg.train.train_path,
        train=True,
        transforms=trans,
        cfg=cfg,
    )

    # 学習用と検証用にデータセットを分割
    # n_samples = len(dataset)
    # train_size = int(n_samples * 0.95)
    # val_size = n_samples - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # val_dataset.train = False

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=cfg.train.batch_size,   
    #     shuffle=True,
    #     num_workers=cfg.train.num_workers,      
    #     pin_memory=False,
    #     drop_last=True,
    #     collate_fn=None,
    # )
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=cfg.train.num_workers,      
    #     pin_memory=False,
    #     drop_last=True,
    #     collate_fn=None,
    # )
    # return train_loader, val_loader, dataset
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )
    return train_loader, dataset


def make_test_loader(cfg):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    dataset = KablabDataset(
        data_root=cfg.train.test_path,
        train=False,
        transforms=trans,
        cfg=cfg,
    )
    test_loader = DataLoader(
        dataset=dataset,
        # batch_size=cfg.train.batch_size,
        batch_size=1,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, dataset


def train_one_epoch_with_d(model: nn.Module, discriminator, train_loader, optimizer, optimizer_d, loss_f_mse, loss_f_train, device, cfg):
    epoch_loss = 0
    epoch_loss_d = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()
    discriminator.train()
    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        (lip, target, feat_add), data_len, label = batch
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
            loss_d = loss_f_train.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="d")
        except Exception:
            print("error")
        
        loss_d.backward()
        # gradient clipping
        clip_grad_norm_(discriminator.parameters(), cfg.train.max_norm)
        optimizer_d.step()
        epoch_loss_d += loss_d.item()

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
            loss_g_ls = loss_f_train.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="g")
            loss_g_fm = loss_f_train.fm_loss(fmaps_f, fmaps_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d)
        except Exception:
            print("error")

        # loss_default = loss_f_mse(output[:, :, :-2], target[:, :, 2:]) + loss_f_mse(dec_output[:, :, :-2], target[:, :, 2:])
        # loss_mask = loss_f_train.masked_mse(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
        #     + loss_f_train.masked_mse(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # loss_delta = loss_f_train.delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 

        # postnet前後のmse_lossと、GAN関連のlossの和（実際は重みづけが必要そう）
        loss = loss_f_train.mse_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + loss_f_train.mse_loss(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                + loss_f_train.delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                    + loss_g_ls + loss_g_fm
        
        loss.backward()
        # gradient clipping
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        epoch_loss += loss.item()

        wandb.log({"train_iter_loss_disc": loss_d.item()})
        wandb.log({"train_iter_loss_gen": loss.item()})
    
    # epoch_loss /= data_cnt
    # epoch_loss_d /= data_cnt
    epoch_loss /= iter_cnt
    epoch_loss_d /= iter_cnt
    return epoch_loss, epoch_loss_d


def calc_test_loss_with_d(model: nn.Module, discriminator, test_loader, loss_f_val, device, cfg):
    epoch_loss = 0
    epoch_loss_d = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(test_loader)
    print("calc test loss")
    model.eval()
    discriminator.eval()
    for batch in test_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        (lip, target, feat_add), data_len, label = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)
        batch_size = lip.shape[0]
        data_cnt += batch_size

        # generatorでデータ生成
        with torch.no_grad():
            output, dec_output = model.inference(
                lip=lip,
            )    
    
        # discriminatorへ入力
        try:
            out_f, fmaps_f = discriminator(output)   # 生成データを入力
            out_r, fmaps_r = discriminator(target)    # 実データを入力
            loss_d = loss_f_val.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="d")
            loss_g_ls = loss_f_val.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="g")
            loss_g_fm = loss_f_val.fm_loss(fmaps_f, fmaps_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d)
        except Exception:
            print("error")

        loss = loss_f_val.mse_loss(output, target, data_len, max_len=model.max_len * 2) \
            + loss_f_val.mse_loss(dec_output, target, data_len, max_len=model.max_len * 2) \
                + loss_f_val.delta_loss(output, target, data_len, max_len=model.max_len * 2) \
                    + loss_g_ls + loss_g_fm

        epoch_loss += loss.item()
        epoch_loss_d += loss_d.item()

        wandb.log({"val_iter_loss_disc": loss_d.item()})
        wandb.log({"val_iter_loss_gen": loss.item()})
    
    # epoch_loss /= data_cnt
    # epoch_loss_d /= data_cnt
    epoch_loss /= iter_cnt
    epoch_loss_d /= iter_cnt
    return epoch_loss, epoch_loss_d


def save_result(loss_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(save_path)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    #resultの表示
    result_path = 'results'
    os.makedirs(result_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = os.path.join(cfg.train.train_save_path, current_time)
    os.makedirs(save_path, exist_ok=True)
    
    # Dataloader作成
    # train_loader, val_loader, _ = make_train_val_loader(cfg)
    train_loader, _ = make_train_val_loader(cfg)
    test_loader, _ = make_test_loader(cfg)
    
    # 損失関数
    loss_f_mse = nn.MSELoss()
    loss_f_mae = nn.L1Loss()
    loss_f_train = masked_loss(train=True)
    loss_f_val = masked_loss(train=False)
    train_loss_list = []
    
    with wandb.init(**cfg.wandb.setup, config=wandb_cfg) as run:
        # model
        model = Lip2SP(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            res_layers=cfg.model.res_layers,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_head=cfg.model.n_head,
            glu_inner_channels=cfg.model.d_model,
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
        ).to(device)

        # discriminator
        assert cfg.model.which_d == "jcu" or "unet", print('discriminatorを設定してください!')
        if cfg.model.which_d == "unet":
            discriminator = UNetDiscriminator()
        elif cfg.model.which_d == "jcu":
            discriminator = JCUDiscriminator(
                in_channels=cfg.model.out_channels,
                out_channels=1,
                use_gc=cfg.train.use_gc,
                emb_in=cfg.train.batch_size,
            )
        discriminator.to(device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay     
        )

        # schedular
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.train.max_epoch // 4, 
            gamma=cfg.train.lr_decay_rate      
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(
            optimizer_d, 
            step_size=cfg.train.max_epoch // 4, 
            gamma=cfg.train.lr_decay_rate      
        )

        wandb.watch(model, **cfg.wandb.watch)
        wandb.watch(discriminator, **cfg.wandb.watch)
            
        # training
        for epoch in range(cfg.train.max_epoch):
            print(f"##### {epoch} #####")
            epoch_loss, epoch_loss_d = train_one_epoch_with_d(model, discriminator, train_loader, optimizer, optimizer_d, loss_f_mse, loss_f_train, device, cfg)
            train_loss_list.append(epoch_loss)
            print(f"epoch_loss = {epoch_loss}")
            print(f"epoch_loss_d = {epoch_loss_d}")
            print(f"train_loss_list = {train_loss_list}")

            if epoch % cfg.train.display_test_loss_step == 0:
                epoch_loss_test, epoch_loss_d_test = calc_test_loss_with_d(model, discriminator, test_loader, loss_f_val, device, cfg)
                print(f"epoch_loss_test = {epoch_loss_test}")
                print(f"epoch_loss_d_test = {epoch_loss_d_test}")
            
            # 学習率の更新
            scheduler.step()
            scheduler_d.step()

        # モデルの保存
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{cfg.model.name}_d.pth"))
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(save_path, f"model_{cfg.model.name}_d.pth"))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__=='__main__':
    main()