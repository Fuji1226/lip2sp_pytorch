"""
discriminatorを使う場合のtrain
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow
import wandb

from pathlib import Path
from tqdm import tqdm
import os
import time
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# 自作
from loss import MaskedLoss, AdversarialLoss
from model.discriminator import JCUDiscriminator, UNetDiscriminator
from train_default import make_train_val_loader, make_model


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def save_checkpoint(model, discriminator, optimizer, optimizer_d, schedular, schedular_d, epoch, ckpt_path):
	torch.save({'model': model.state_dict(),
                'discriminator': discriminator.state_dict(),
				'optimizer': optimizer.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'schedular': schedular.state_dict(),
                'schedular_d': schedular_d.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), 
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                'cuda_random' : torch.cuda.get_rng_state(),
				'epoch': epoch}, ckpt_path)


def make_discriminator(cfg, device):
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
    return discriminator.to(device)


def train_one_epoch_with_d(model, discriminator, train_loader, optimizer, optimizer_d, loss_f, loss_f_adv, device, cfg, training_method, mixing_prob):
    epoch_loss = 0
    epoch_loss_d = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)

    print("iter start")
    model.train()
    discriminator.train()
    rf = cfg.model.reduction_factor      

    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        batch_size = lip.shape[0]
        data_cnt += batch_size

        #====================================================
        # discriminatorの最適化
        #====================================================
        # discriminatorのパラメータ更新を行うように設定
        for param in discriminator.parameters():
            param.requires_grad = True

        # output : postnet後の出力
        # dec_output : postnet前の出力
        # generatorのパラメータが更新されないように設定
        with torch.no_grad():
            output, dec_output, enc_output = model(
                lip=lip,
                prev=feature,
                data_len=data_len,
                training_method=training_method,
                mixing_prob=mixing_prob,
            )                      
        
        # discriminatorへ入力
        out_f, fmaps_f = discriminator(output)   # 生成データを入力
        out_r, fmaps_r = discriminator(feature)    # 実データを入力

        # 損失計算
        loss_d = loss_f_adv.ls_loss(out_f, out_r, which_d=cfg.model.which_d, which_loss="d", data_len=data_len, max_len=model.max_len * 2)

        # 勾配初期化
        optimizer_d.zero_grad()
        loss_d.backward()

        # gradient clipping
        clip_grad_norm_(discriminator.parameters(), cfg.train.max_norm)

        optimizer_d.step()
        epoch_loss_d += loss_d.item()
        wandb.log({"train_loss_disc": loss_d.item()})

        #====================================================
        # generatorの最適化
        #====================================================
        # discriminatorのパラメータ更新を行わないように設定
        for param in discriminator.parameters():
            param.requires_grad = False
        
        # generatorでデータ生成
        output, dec_output, enc_output = model(
            lip=lip,
            prev=feature,
            data_len=data_len,
            training_method=training_method,
            mixing_prob=mixing_prob,
        )                      

        # discriminatorへ入力
        out_f, fmaps_f = discriminator(output)   # 生成データを入力
        out_r, fmaps_r = discriminator(feature)    # 実データを入力
        loss_g_ls = loss_f_adv.ls_loss(out_f, out_r, which_d=cfg.model.which_d, which_loss="g", data_len=data_len, max_len=model.max_len * 2)
        loss_g_fm = loss_f_adv.fm_loss(fmaps_f, fmaps_r, which_d=cfg.model.which_d, data_len=data_len, max_len=model.max_len * 2)

        # generatorのloss
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=model.max_len * 2)
        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=model.max_len * 2)
        delta_loss = loss_f.delta_loss(output, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur, batch_norm=cfg.train.batch_norm)
        loss_recon = output_loss + dec_output_loss + delta_loss

        # GANspeechのscaled feature matching lossを使用
        co_fm = loss_recon / loss_g_fm

        # total loss
        loss = loss_recon + loss_g_ls + co_fm * loss_g_fm

        # 勾配初期化
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        epoch_loss += loss.item()

        wandb.log({"train_gen_ls_loss": loss_g_ls.item()})
        wandb.log({"train_gen_fm_loss": loss_g_fm.item()})
        wandb.log({"train_output_loss": output_loss.item()})
        wandb.log({"train_dec_output_loss": dec_output_loss.item()})
        wandb.log({"train_delta_loss": delta_loss.item()})
        wandb.log({"train_loss_gen": loss.item()})
    
    epoch_loss /= iter_cnt
    epoch_loss_d /= iter_cnt
    return epoch_loss, epoch_loss_d


def calc_test_loss_with_d(model: nn.Module, discriminator, val_loader, loss_f, loss_f_adv, device, cfg):
    epoch_loss = 0
    epoch_loss_d = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()
    discriminator.eval()
    rf = cfg.model.reduction_factor      

    for batch in val_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        batch_size = lip.shape[0]
        data_cnt += batch_size

        with torch.no_grad():
            output, dec_output, enc_output = model(lip)
            out_f, fmaps_f = discriminator(output)   
            out_r, fmaps_r = discriminator(feature)    

        loss_d = loss_f_adv.ls_loss(out_f, out_r, which_d=cfg.model.which_d, which_loss="d", data_len=data_len, max_len=model.max_len * 2)
        loss_g_ls = loss_f_adv.ls_loss(out_f, out_r, which_d=cfg.model.which_d, which_loss="g", data_len=data_len, max_len=model.max_len * 2)
        loss_g_fm = loss_f_adv.fm_loss(fmaps_f, fmaps_r, which_d=cfg.model.which_d, data_len=data_len, max_len=model.max_len * 2)

        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=model.max_len * 2)
        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=model.max_len * 2)
        delta_loss = loss_f.delta_loss(output, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur, batch_norm=cfg.train.batch_norm)

        loss = output_loss + dec_output_loss + delta_loss + loss_g_ls + loss_g_fm

        epoch_loss += loss.item()
        epoch_loss_d += loss_d.item()

        wandb.log({"val_gen_ls_loss": loss_g_ls.item()})
        wandb.log({"val_gen_fm_loss": loss_g_fm.item()})
        wandb.log({"val_output_loss": output_loss.item()})
        wandb.log({"val_dec_output_loss": dec_output_loss.item()})
        wandb.log({"val_delta_loss": delta_loss.item()})
        wandb.log({"val_loss_gen": loss.item()})
    
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

    print(f"cpu_num = {os.cpu_count()}")
    torch.backends.cudnn.benchmark = True

    # 口唇動画か顔かの選択
    lip_or_face = cfg.train.face_or_lip
    assert lip_or_face == "face" or "lip"
    if lip_or_face == "face":
        data_path = cfg.train.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    elif lip_or_face == "lip":
        data_path = cfg.train.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    
    print("--- data directory check ---")
    print(f"data_path = {data_path}")
    print(f"mean_std_path = {mean_std_path}")

    # check pointの保存先を指定
    ckpt_path = os.path.join(cfg.train.ckpt_path, lip_or_face, current_time)
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = os.path.join(cfg.train.train_save_path, lip_or_face, current_time)
    os.makedirs(save_path, exist_ok=True)
    
    # Dataloader作成
    train_loader, val_loader, _ = make_train_val_loader(cfg, data_path, mean_std_path)
    
    # 損失関数
    loss_f = MaskedLoss(train=True)
    loss_f_adv = AdversarialLoss(train=True)
    train_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model =make_model(cfg, device)
        model_path = cfg.train.model_path
        model.load_state_dict(torch.load(model_path))

        # discriminator
        discriminator = make_discriminator(cfg, device)
        
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=cfg.train.multi_lr_decay_step,
            gamma=cfg.train.lr_decay_rate      
        )
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_d, 
            milestones=cfg.train.multi_lr_decay_step,
            gamma=cfg.train.lr_decay_rate      
        )

        if cfg.train.check_point_start:
            checkpoint_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/2022:06:24_10-36-39/mspec_40.ckpt"
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            scheduler.load_state_dict(checkpoint["schedular"])
            scheduler_d.load_state_dict(checkpoint["schedular_d"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])

        wandb.watch(model, **cfg.wandb_conf.watch)
        wandb.watch(discriminator, **cfg.wandb_conf.watch)

        if cfg.train.debug:
            max_epoch = cfg.train.debug_max_epoch
        else:
            max_epoch = cfg.train.max_epoch
            
        training_method = cfg.train.training_method
        mixing_prob = cfg.train.mixing_prob

        # training
        for epoch in range(max_epoch):
            print(f"##### {epoch} #####")
            print(f"training_method : {training_method}")
            print(f"mixing_prob = {mixing_prob}")
            print(f"learning_rate = {scheduler.get_last_lr()}")
            
            # training
            epoch_loss, epoch_loss_d = train_one_epoch_with_d(
                model=model, 
                discriminator=discriminator, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                optimizer_d=optimizer_d, 
                loss_f=loss_f, 
                loss_f_adv=loss_f_adv,
                device=device, 
                cfg=cfg,
                training_method=training_method,
                mixing_prob=mixing_prob,
            )
            train_loss_list.append(epoch_loss)
            print(f"epoch_loss = {epoch_loss}")
            print(f"epoch_loss_d = {epoch_loss_d}")
            print(f"train_loss_list = {train_loss_list}")

            # validation
            if epoch % cfg.train.display_val_loss_step == 0:
                epoch_loss_test, epoch_loss_d_test = calc_test_loss_with_d(
                    model=model, 
                    discriminator=discriminator, 
                    val_loader=val_loader, 
                    loss_f=loss_f, 
                    loss_f_adv=loss_f_adv,
                    device=device, 
                    cfg=cfg,
                )
                print(f"epoch_loss_test = {epoch_loss_test}")
                print(f"epoch_loss_d_test = {epoch_loss_d_test}")
            
            # 学習率の更新
            scheduler.step()
            scheduler_d.step()

            # check point
            if epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    discriminator=discriminator,
                    optimizer=optimizer,
                    optimizer_d=optimizer_d,
                    schedular=scheduler,
                    schedular_d=scheduler_d,
                    epoch=epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_d_{epoch}.ckpt")
                )

        # モデルの保存
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{cfg.model.name}_d.pth"))
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(save_path, f"model_{cfg.model.name}_d.pth"))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__=='__main__':
    main()