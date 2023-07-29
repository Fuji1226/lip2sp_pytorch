from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from timm.scheduler.cosine_lr import CosineLRScheduler

from utils import make_train_val_loader_tts, save_loss, get_path_train, check_mel_nar, count_params, set_config
from model.model_asr import SpeechRecognizer
from data_process.phoneme_encode import IGNORE_INDEX

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    model, optimizer, scheduler, scaler,
    train_loss_list,
    val_loss_list,
    epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
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
    model = SpeechRecognizer(
        in_channels=cfg.model.n_mel_channels,
        hidden_channels=cfg.model.asr_hidden_channels,
        n_conv_layers=cfg.model.asr_n_conv_layers,
        conv_dropout=cfg.model.asr_conv_dropout,
        trans_enc_n_layers=cfg.model.asr_trans_enc_n_layers,
        trans_enc_n_head=cfg.model.asr_trans_enc_n_head,
        trans_dec_n_layers=cfg.model.asr_trans_dec_n_layers,
        trans_dec_n_head=cfg.model.asr_trans_dec_n_head,
        out_channels=cfg.model.asr_out_channels,
    )

    count_params(model, "model")
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, scaler, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        feature = feature.to(device)
        text = text.to(device)
        feature_len = feature_len.to(device)
        text_len = text_len.to(device)

        text_input = text[:, :-1]
        text_output = text[:, 1:]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(feature, feature_len, text_input)
            loss = loss_f(output, text_output)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        wandb.log({"train_loss": loss})

        iter_cnt += 1

        if cfg.train.debug and iter_cnt > cfg.train.debug_iter:
            break
        
    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("\ncalc val loss")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        feature = feature.to(device)
        text = text.to(device)
        feature_len = feature_len.to(device)
        text_len = text_len.to(device)

        text_input = text[:, :-1]
        text_output = text[:, 1:]
        
        with torch.no_grad():
            output = model(feature, feature_len, text_input)
            loss = loss_f(output, text_output)

        epoch_loss += loss.item()
        wandb.log({"val_loss": loss})

        iter_cnt += 1

        if cfg.train.debug and iter_cnt > cfg.train.debug_iter:
            break
        
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
    train_loader, val_loader, _, _ = make_train_val_loader_tts(cfg, train_data_root, val_data_root)

    # finetuning
    if cfg.train.finetuning:
        assert len(cfg.train.speaker) == 1
        print(f"finetuning {cfg.train.speaker}")
        cfg.train.speaker = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab"]

    loss_f = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    train_loss_list = []
    val_loss_list = []
    
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        if cfg.train.which_optim == 'adam':
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )
        elif cfg.train.which_optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=cfg.train.lr,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )

        if cfg.train.which_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=cfg.train.lr_decay_exp,
            )
        elif cfg.train.which_scheduler == 'warmup':
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=cfg.train.max_epoch,
                lr_min=cfg.train.warmup_lr_min,
                warmup_t=int(cfg.train.max_epoch * cfg.train.warmup_t_rate),
                warmup_lr_init=cfg.train.warmup_lr_init,
                warmup_prefix=True,
            )

        scaler = torch.cuda.amp.GradScaler()
        
        last_epoch = 0
        if cfg.train.check_point_start:
            print("load check point")
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_loss_list = checkpoint["train_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]
            
        if cfg.train.check_point_start_separate_save_dir:
            print("load check point (separate save dir)")
            checkpoint_path = Path(cfg.train.start_ckpt_path_separate_save_dir).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model"])
            cfg.train.lr = 0.0001

        wandb.watch(model, **cfg.wandb_conf.watch)
    
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            # training
            epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                scaler=scaler,
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)

            # validation
            epoch_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)

            if cfg.train.which_scheduler == 'exp':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == 'warmup':
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
                scheduler.step(current_epoch)

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    val_loss_list=val_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
                
        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()