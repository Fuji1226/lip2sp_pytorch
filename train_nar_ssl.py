from omegaconf import OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
import torch
from timm.scheduler import CosineLRScheduler

from utils import (
    count_params,
    set_config,
    save_loss,
    check_mel_nar,
    fix_random_seed,
    get_path_train_raw,
    make_train_val_loader_with_external_data_raw,
)
from model.model_nar_ssl import Lip2SpeechSSL, Lip2SpeechLightWeight
from loss import MaskedLoss

wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        train_loss_list,
        train_mae_loss_list,
        train_mse_loss_list,
        val_loss_list,
        val_mae_loss_list,
        val_mse_loss_list,
        epoch,
        ckpt_path,
):
    torch.save(
        {
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
            'train_mae_loss_list': train_mae_loss_list,
            'train_mse_loss_list': train_mse_loss_list,
            'val_loss_list': val_loss_list,
            'val_mae_loss_list': val_mae_loss_list,
            'val_mse_loss_list': val_mse_loss_list,
            'epoch': epoch,
        }, 
        ckpt_path,
    )


def make_model(
        cfg,
        device,
):
    if cfg.model.model_name == 'lightweight':
        model = Lip2SpeechLightWeight(cfg)
    else:
        model = Lip2SpeechSSL(cfg)
    count_params(model, 'model')
    return model.to(device)


def train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        loss_f,
        device,
        cfg,
        ckpt_time,
):
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_mse_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("training") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(
                lip=lip,
                audio=None,
                lip_len=lip_len,
                spk_emb=spk_emb,
            )
            mae_loss = loss_f.mae_loss(output, feature, feature_len, max_len=output.shape[-1])
            mse_loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])
            loss = mae_loss
            epoch_mae_loss += mae_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_loss += loss.item()
            wandb.log({"train_mae_loss": mae_loss})
            wandb.log({"train_mse_loss": mse_loss})
            wandb.log({"train_loss": loss})
            loss = loss / cfg.train.iters_to_accumulate

        scaler.scale(loss).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_mae_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    return epoch_loss, epoch_mae_loss, epoch_mse_loss


def val_one_epoch(
        model,
        val_loader,
        loss_f,
        device,
        cfg,
        ckpt_time,
):
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_mse_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("validation")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                output = model(
                    lip=lip,
                    audio=None,
                    lip_len=lip_len,
                    spk_emb=spk_emb,
                )
            mae_loss = loss_f.mae_loss(output, feature, feature_len, max_len=output.shape[-1])
            mse_loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])
            loss = mae_loss
            epoch_mae_loss += mae_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_loss += loss.item()
            wandb.log({"val_mae_loss": mae_loss})
            wandb.log({"val_mse_loss": mse_loss})
            wandb.log({"val_loss": loss})
            
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
                break

        if all_iter - 1 > 0:
            if iter_cnt % (all_iter - 1) == 0:
                check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
        else:
            check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    epoch_mae_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    return epoch_loss, epoch_mae_loss, epoch_mse_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
    fix_random_seed(cfg.train.random_seed)
    
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")

    video_dir, audio_dir, ckpt_path, save_path, ckpt_time= get_path_train_raw(cfg, current_time)
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_with_external_data_raw(cfg, video_dir, audio_dir)
    
    loss_f = MaskedLoss()
    train_loss_list = []
    train_mae_loss_list = []
    train_mse_loss_list = []
    val_loss_list = []
    val_mae_loss_list = []
    val_mse_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
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

        if cfg.train.check_point_start_separate_save_dir:
            print("load check point (separate save dir)")
            checkpoint_path = Path(cfg.train.start_ckpt_path_separate_save_dir).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model"])

        wandb.watch(model, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            epoch_loss, epoch_mae_loss, epoch_mse_loss = train_one_epoch( 
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
            train_mae_loss_list.append(epoch_mae_loss)
            train_mse_loss_list.append(epoch_mse_loss)

            epoch_loss, epoch_mae_loss, epoch_mse_loss = val_one_epoch(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_mae_loss_list.append(epoch_mae_loss)
            val_mse_loss_list.append(epoch_mse_loss)

            if cfg.train.which_scheduler == 'exp':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == 'warmup':
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
                scheduler.step(epoch)
            
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    train_mae_loss_list=train_mae_loss_list,
                    train_mse_loss_list=train_mse_loss_list,
                    val_loss_list=val_loss_list,
                    val_mae_loss_list=val_mae_loss_list,
                    val_mse_loss_list=val_mse_loss_list,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{current_epoch}.ckpt")
                )

            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_mae_loss_list, val_mae_loss_list, save_path, "mae_loss")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            
    wandb.finish()
    
    
if __name__ == "__main__":
    main()