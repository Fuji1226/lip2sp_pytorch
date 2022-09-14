"""
enhancerを取り入れた学習用
モデルは事前学習済みのものを読み込み,合成音声をenhancerの学習データとする
モデルは学習せず,enhancerのみを学習する
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly

# 自作
from model.dataset_npz import KablabDataset, KablabTransform, KablabTransform_val,  MySubset, collate_fn_padding
from model.enhancer import Enhancer1D, Enhancer2D
from model.model_default import Lip2SP
from loss import MaskedLoss
from train_default import make_train_val_loader, make_model

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
random.seed(7)


def save_checkpoint(enhancer, optimizer, schedular, epoch, ckpt_path):
	torch.save({'enhancer': enhancer.state_dict(),
				'optimizer': optimizer.state_dict(),
                'schedular': schedular.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), 
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                'cuda_random' : torch.cuda.get_rng_state(),
				'epoch': epoch}, ckpt_path)


def save_result(loss_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(save_path)


def make_enhancer(cfg, device):
    if cfg.model.which_enhancer == "2D":
        enhancer = Enhancer2D(
            in_channels=1,
            out_channels=1,
            use_lstm=cfg.train.use_lstm,
            lstm_layers=cfg.model.lstm_layers,
            bidirectional=cfg.model.bidirectional,
        )
    elif cfg.model.which_enhancer == "1D":
        enhancer = Enhancer1D(
            in_channels=cfg.model.out_channels,
            out_channels=cfg.model.out_channels,
            lstm_layers=cfg.model.lstm_layers,
            bidirectional=cfg.model.bidirectional,
        )
    return enhancer.to(device)


def train_one_epoch(model, enhancer, train_loader, optimizer, loss_f, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    rf = cfg.model.reduction_factor
    model.eval()
    enhancer.train()

    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        # 口唇動画から音響特徴量を推定
        with torch.no_grad():
            if cfg.train.auto_regressive:
                output, dec_output, enc_output = model(lip)
            else:
                output, dec_output, enc_output = model(
                    lip=lip,
                    prev=feature,
                    data_len=data_len,
                    training_method=cfg.train.training_method,
                    mixing_prob=cfg.train.mixing_prob,
                )               

        # 推定結果をenhancerに入力
        enhanced_dec_output = enhancer(dec_output)
        enhanced_output = enhancer(output)
        enhanced_feature = enhancer(feature)

        dec_output_loss = loss_f.mse_loss(enhanced_dec_output, feature, data_len, max_len=model.max_len * rf)
        output_loss = loss_f.mse_loss(enhanced_output, feature, data_len, max_len=model.max_len * rf)
        delta_loss = loss_f.delta_loss(enhanced_output, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur, batch_norm=cfg.train.batch_norm)
        output_loss_feature = loss_f.mse_loss(enhanced_feature, feature, data_len, max_len=model.max_len * rf)
        delta_loss_feature = loss_f.delta_loss(enhanced_feature, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur, batch_norm=cfg.train.batch_norm)

        loss = dec_output_loss + output_loss + delta_loss + output_loss_feature + delta_loss_feature

        # 勾配の初期化
        optimizer.zero_grad()

        loss.backward()

        # gradient clipping
        clip_grad_norm_(enhancer.parameters(), cfg.train.max_norm)

        optimizer.step()
        epoch_loss += loss.item()
        
        wandb.log({"train_dec_output_loss": dec_output_loss.item()})
        wandb.log({"train_output_loss": output_loss.item()})
        wandb.log({"train_delta_loss": delta_loss.item()})
        wandb.log({"train_output_loss_feature": output_loss_feature.item()})
        wandb.log({"train_delta_loss_feature": delta_loss_feature.item()})
        wandb.log({"train_total_loss": loss.item()})

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                break

    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model, enhancer, val_loader, loss_f, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()
    enhancer.eval()
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
            # enhanced_dec_output = enhancer(dec_output)
            enhanced_output = enhancer(output)
            enhanced_feature = enhancer(feature)

        # dec_output_loss = loss_f.mse_loss(enhanced_dec_output, feature, data_len, max_len=model.max_len * rf)
        output_loss = loss_f.mse_loss(enhanced_output, feature, data_len, max_len=model.max_len * rf)
        delta_loss = loss_f.delta_loss(enhanced_output, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur, batch_norm=cfg.train.batch_norm)
        output_loss_feature = loss_f.mse_loss(enhanced_feature, feature, data_len, max_len=model.max_len * rf)
        delta_loss_feature = loss_f.delta_loss(enhanced_feature, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur, batch_norm=cfg.train.batch_norm)

        # loss = dec_output_loss +  output_loss + delta_loss + output_loss_feature + delta_loss_feature
        loss = output_loss + delta_loss + output_loss_feature + delta_loss_feature

        epoch_loss += loss.item()

        # wandb.log({"val_dec_output_loss": dec_output_loss.item()})
        wandb.log({"val_output_loss": output_loss.item()})
        wandb.log({"val_delta_loss": delta_loss.item()})
        wandb.log({"val_output_loss_feature": output_loss_feature.item()})
        wandb.log({"val_delta_loss_feature": delta_loss_feature.item()})
        wandb.log({"val_total_loss": loss.item()})

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                break
    
    epoch_loss /= iter_cnt
    return epoch_loss


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
    elif lip_or_face == "lip_128128":
        data_path = cfg.train.lip_pre_loaded_path_128128
        mean_std_path = cfg.train.lip_mean_std_path_128128

    print("--- data directory check ---")
    print(f"data_path = {data_path}")
    print(f"mean_std_path = {mean_std_path}")

    # check point
    ckpt_path = os.path.join(cfg.train.ckpt_path, lip_or_face, current_time)
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = os.path.join(cfg.train.train_save_path, lip_or_face, current_time)
    os.makedirs(save_path, exist_ok=True)
    
    # Dataloader作成
    train_loader, val_loader, _ = make_train_val_loader(cfg, data_path, mean_std_path)
    
    # 損失関数
    loss_f = MaskedLoss(train=True)
    train_loss_list = []
    
    # training
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg) as run:
        # model(事前学習済みモデルをロード)
        model = make_model(cfg, device)
        model_path = cfg.train.model_path
        model.load_state_dict(torch.load(model_path))

        # enhancer
        enhancer = make_enhancer(cfg, device)
    
        # optimizer
        optimizer = torch.optim.Adam(
            enhancer.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.train.multi_lr_decay_step,
            gamma=cfg.train.lr_decay_rate,
        )

        if cfg.train.check_point_start:
            checkpoint_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/2022:06:24_10-36-39/mspec_40.ckpt"
            checkpoint = torch.load(checkpoint_path)
            enhancer.load_state_dict(checkpoint["enhancer"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["schedular"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])

        wandb.watch(enhancer, **cfg.wandb_conf.watch)

        if cfg.train.debug:
            max_epoch = cfg.train.debug_max_epoch
        else:
            max_epoch = cfg.train.max_epoch
        
        for epoch in range(max_epoch):
            print(f"##### {epoch} #####")
            print(f"learning_rate = {scheduler.get_last_lr()}")

            epoch_loss = train_one_epoch(
                model=model, 
                enhancer=enhancer,
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
            )
            train_loss_list.append(epoch_loss)
            print(f"epoch_loss = {epoch_loss}")
            print(f"train_loss_list = {train_loss_list}")

            # 検証用データ
            if epoch % cfg.train.display_val_loss_step == 0:
                epoch_loss_test = calc_val_loss(
                    model=model, 
                    enhancer=enhancer,
                    val_loader=val_loader, 
                    loss_f=loss_f, 
                    device=device, 
                    cfg=cfg,
                )
                print(f"epoch_loss_test = {epoch_loss_test}")
            
            # 学習率の更新
            scheduler.step()

            # check point
            if epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    enhancer=enhancer,
                    optimizer=optimizer,
                    schedular=scheduler,
                    epoch=epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.ckpt")
                )
                
        # モデルの保存
        torch.save(enhancer.state_dict(), os.path.join(save_path, f"enhancer_{cfg.model.name}.pth"))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(os.path.join(save_path, f"enhancer_{cfg.model.name}.pth"))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()