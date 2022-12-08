from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from librosa.display import specshow
from itertools import chain

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import make_train_val_loader_ssl, save_loss, get_path_train, count_params, set_config, calc_class_balance, check_movie
from model.model_nar_separate import Encoder, VideoDecoder, MelDecoder
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    encoder, video_decoder, optimizer, scheduler,
    train_loss_list,
    val_loss_list,
    epoch, ckpt_path):
	torch.save({
        'encoder': encoder.state_dict(),
        'video_decoder': video_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    encoder = Encoder(
        in_channels=cfg.model.in_channels,
        res_inner_channels=cfg.model.res_inner_channels,
        which_res=cfg.model.which_res,
        rnn_n_layers=cfg.model.rnn_n_layers,
        trans_n_layers=cfg.model.trans_enc_n_layers,
        trans_n_head=cfg.model.trans_enc_n_head,
        which_encoder=cfg.model.which_encoder,
        res_dropout=cfg.train.res_dropout,
        rnn_dropout=cfg.train.rnn_dropout,
        reduction_factor=cfg.model.reduction_factor,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
    )
    video_decoder = VideoDecoder(
        in_channels=int(cfg.model.res_inner_channels * 8),
        out_channels=cfg.model.in_channels,
    )

    count_params(encoder, "encoder")
    count_params(video_decoder, "video_decoder")
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return encoder.to(device), video_decoder.to(device)


def train_one_epoch(encoder, video_decoder, train_dataset, train_loader, optimizer, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    encoder.train()
    video_decoder.train()

    lip_mean = train_dataset.lip_mean
    lip_std = train_dataset.lip_std

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, lip_mask, upsample, data_len, speaker, label = batch
        lip = lip.to(device)
        lip_mask = lip_mask.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        data_len = torch.div(data_len, cfg.model.reduction_factor).to(dtype=torch.int)

        enc_output = encoder(lip_mask, data_len)
        output = video_decoder(enc_output)

        loss = loss_f.mse_loss(output, lip, data_len, max_len=output.shape[-1])

        epoch_loss += loss.item()
        wandb.log({"train_loss": loss})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_movie(lip[0], output[0], lip_mean, lip_std, cfg, "movie_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_movie(lip[0], output[0], lip_mean, lip_std, cfg, "movie_train", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(encoder, video_decoder, train_dataset, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("\ncalc val loss")
    encoder.eval()
    video_decoder.eval()

    lip_mean = train_dataset.lip_mean
    lip_std = train_dataset.lip_std

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, lip_mask, upsample, data_len, speaker, label = batch
        lip = lip.to(device)
        lip_mask = lip_mask.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        data_len = torch.div(data_len, cfg.model.reduction_factor).to(dtype=torch.int)
        
        with torch.no_grad():
            enc_output = encoder(lip_mask, data_len)
            output = video_decoder(enc_output)

        loss = loss_f.mse_loss(output, lip, data_len, max_len=output.shape[-1])

        epoch_loss += loss.item()
        wandb.log({"val_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_movie(lip[0], output[0], lip_mean, lip_std, cfg, "movie_val", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_movie(lip[0], output[0], lip_mean, lip_std, cfg, "movie_val", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    return epoch_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
        
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # path
    train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"train_data_root = {train_data_root}")
    print(f"val_data_root = {val_data_root}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_ssl(cfg, train_data_root, val_data_root)

    # finetuning
    if cfg.train.finetuning:
        assert len(cfg.train.speaker) == 1
        print(f"finetuning {cfg.train.speaker}")
        cfg.train.speaker = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab"]

    # 損失関数
    if len(cfg.train.speaker) > 1:
        class_weight = calc_class_balance(cfg, train_data_root, device)
    else:
        class_weight = None
    loss_f = MaskedLoss(weight=class_weight, use_weighted_mean=cfg.train.use_weighted_mean)

    train_loss_list = []
    val_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        encoder, video_decoder = make_model(cfg, device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=chain(encoder.parameters(), video_decoder.parameters()),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.train.multi_lr_decay_step,
            gamma=cfg.train.lr_decay_rate,
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            print("load check point")
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            encoder.load_state_dict(checkpoint["encoder"])
            video_decoder.load_state_dict(checkpoint["video_decoder"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_loss_list = checkpoint["train_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]

        wandb.watch(encoder, **cfg.wandb_conf.watch)
        wandb.watch(video_decoder, **cfg.wandb_conf.watch)
    
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            # training
            epoch_loss = train_one_epoch(
                encoder=encoder,
                video_decoder=video_decoder,
                train_dataset=train_dataset,
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)

            # validation
            epoch_loss = calc_val_loss(
                encoder=encoder,
                video_decoder=video_decoder,
                train_dataset=train_dataset,
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
        
            # 学習率の更新
            scheduler.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    encoder=encoder,
                    video_decoder=video_decoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_list=train_loss_list,
                    val_loss_list=val_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
                
        # モデルの保存
        enc_save_path = save_path / f"encoder_{cfg.model.name}.pth"
        torch.save(encoder.state_dict(), str(enc_save_path))
        dec_save_path = save_path / f"video_decoder_{cfg.model.name}.pth"
        torch.save(encoder.state_dict(), str(dec_save_path))
            
    wandb.finish()


if __name__=='__main__':
    main()