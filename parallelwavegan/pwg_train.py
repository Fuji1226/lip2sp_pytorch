from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from utils import (
    count_params,
    get_path_train,
    save_loss,
    make_train_val_loader_with_external_data,
    make_train_val_loader_pwg,
    set_config,
    check_wav,
    requires_grad_change,
    fix_random_seed,
)
from parallelwavegan.model.generator import Generator
from parallelwavegan.model.discriminator import Discriminator, WaveNetLikeDiscriminator
from parallelwavegan.stft_loss import MultiResolutionSTFTLoss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def save_checkpoint(
    gen, 
    disc, 
    optimizer_g, 
    optimizer_d, 
    scheduler_g, 
    scheduler_d,
    scaler_g,
    scaler_d,
    train_epoch_loss_disc_list,
    train_epoch_loss_gen_stft_list,
    train_epoch_loss_gen_gan_list,
    train_epoch_loss_gen_all_list,
    val_epoch_loss_disc_list,
    val_epoch_loss_gen_stft_list,
    val_epoch_loss_gen_gan_list,
    val_epoch_loss_gen_all_list,
    epoch, 
    ckpt_path
):
    torch.save({
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'scheduler_g': scheduler_g.state_dict(),
        'scheduler_d': scheduler_d.state_dict(),
        'scaler_g': scaler_g.state_dict(),
        'scaler_d': scaler_d.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'train_epoch_loss_disc_list': train_epoch_loss_disc_list,
        'train_epoch_loss_gen_stft_list': train_epoch_loss_gen_stft_list,
        'train_epoch_loss_gen_gan_list': train_epoch_loss_gen_gan_list,
        'train_epoch_loss_gen_all_list': train_epoch_loss_gen_all_list,
        'val_epoch_loss_disc_list': val_epoch_loss_disc_list,
        'val_epoch_loss_gen_stft_list': val_epoch_loss_gen_stft_list,
        'val_epoch_loss_gen_gan_list': val_epoch_loss_gen_gan_list,
        'val_epoch_loss_gen_all_list': val_epoch_loss_gen_all_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    gen = Generator(
        in_channels=cfg.model.pwg_in_channels,
        out_channels=cfg.model.pwg_out_channels,
        inner_channels=cfg.model.pwg_gen_inner_channels,
        cond_channels=cfg.model.pwg_cond_channels,
        upsample_scales=cfg.model.pwg_upsample_scales,
        n_layers=cfg.model.pwg_gen_n_layers,
        n_stacks=cfg.model.pwg_gen_n_stacks,
        dropout=cfg.model.pwg_gen_dropout,
        kernel_size=cfg.model.pwg_kernel_size,
        use_weight_norm=cfg.model.pwg_use_weight_norm,
    )
    if cfg.model.pwg_which_disc == "normal":
        disc = Discriminator(
            in_channels=cfg.model.pwg_in_channels,
            out_channels=cfg.model.pwg_out_channels,
            inner_channels=cfg.model.pwg_disc_inner_channels,
            n_layers=cfg.model.pwg_disc_n_layers,
            kernel_size=cfg.model.pwg_kernel_size,
            use_weight_norm=cfg.model.pwg_use_weight_norm,
            dropout=cfg.model.pwg_disc_dropout,
        )
    elif cfg.model.pwg_which_disc == "wavenet":
        disc = WaveNetLikeDiscriminator(
            n_layers=cfg.model.pwg_disc_n_layers_wavenet,
            n_stacks=cfg.model.pwg_disc_n_stacks,
            in_channels=cfg.model.pwg_in_channels,
            inner_channels=cfg.model.pwg_disc_inner_channels,
            out_channels=cfg.model.pwg_out_channels,
            kernel_size=cfg.model.pwg_kernel_size,
            dropout=cfg.model.pwg_disc_dropout,
        )
    count_params(gen, "generator")
    count_params(disc, "discriminator")
    return gen.to(device), disc.to(device)


def train_one_epoch(
    gen,
    train_loader,
    optimizer_g,
    scaler,
    loss_f,
    device,
    cfg,
    ckpt_time
):
    epoch_loss_disc = 0
    epoch_loss_gen_stft = 0
    epoch_loss_gen_gan = 0
    epoch_loss_gen_all = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("training only generator")
    gen.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
            wav_pred = gen(noise, feature)
            loss_gen_stft = loss_f.calc_loss(wav, wav_pred)
            epoch_loss_gen_stft += loss_gen_stft.item()
            wandb.log({"train_loss_gen_stft": loss_gen_stft})

        scaler.scale(loss_gen_stft).backward()
        scaler.step(optimizer_g)
        scaler.update()
        optimizer_g.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_wav(wav[0], wav_pred[0], cfg, "mel_train", "wav_train_target", "wav_train_output", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_wav(wav[0], wav_pred[0], cfg, "mel_train", "wav_train_target", "wav_train_output", current_time, ckpt_time)

    epoch_loss_disc /= iter_cnt
    epoch_loss_gen_stft /= iter_cnt
    epoch_loss_gen_gan /= iter_cnt
    epoch_loss_gen_all /= iter_cnt
    return epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all


def val_one_epoch(gen, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss_disc = 0
    epoch_loss_gen_stft = 0
    epoch_loss_gen_gan = 0
    epoch_loss_gen_all = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("validation only generator")
    gen.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
                wav_pred = gen(noise, feature)
                loss_gen_stft = loss_f.calc_loss(wav, wav_pred)
                epoch_loss_gen_stft += loss_gen_stft.item()
                wandb.log({"val_loss_gen_stft": loss_gen_stft})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_wav(wav[0], wav_pred[0], cfg, "mel_validation", "wav_validation_target", "wav_validation_output", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_wav(wav[0], wav_pred[0], cfg, "mel_validation", "wav_validation_target", "wav_validation_output", current_time, ckpt_time)

    epoch_loss_disc /= iter_cnt
    epoch_loss_gen_stft /= iter_cnt
    epoch_loss_gen_gan /= iter_cnt
    epoch_loss_gen_all /= iter_cnt
    return epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all


def train_one_epoch_gan(
    gen,
    disc,
    train_loader,
    optimizer_g,
    optimizer_d,
    scaler_g,
    scaler_d,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    epoch_loss_disc = 0
    epoch_loss_gen_stft = 0
    epoch_loss_gen_gan = 0
    epoch_loss_gen_all = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("training gan")
    gen.train()
    disc.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
            wav_pred = gen(noise, feature)

        ### discriminator ###
        disc = requires_grad_change(disc, True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out_real = disc(wav)
            out_pred = disc(wav_pred.detach())
            loss_disc = torch.mean((out_real - 1) ** 2) + torch.mean(out_pred ** 2)
            epoch_loss_disc += loss_disc.item()
            wandb.log({"train_loss_disc": loss_disc})
    
        scaler_d.scale(loss_disc).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()
        optimizer_d.zero_grad()

        ### generator ###
        disc = requires_grad_change(disc, False)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out_pred = disc(wav_pred)
            loss_gen_stft = loss_f.calc_loss(wav, wav_pred)
            loss_gen_gan = torch.mean((out_pred - 1) ** 2)
            loss_gen_all =  cfg.train.stft_loss_weight * loss_gen_stft + cfg.train.gan_loss_weight * loss_gen_gan
            epoch_loss_gen_stft += loss_gen_stft.item()
            epoch_loss_gen_gan += loss_gen_gan.item()
            epoch_loss_gen_all += loss_gen_all.item()
            wandb.log({"train_loss_gen_stft": loss_gen_stft})
            wandb.log({"train_loss_gen_gan": loss_gen_gan})
            wandb.log({"train_loss_gen_all": loss_gen_all})

        scaler_g.scale(loss_gen_all).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        optimizer_g.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_wav(wav[0].to(torch.float32), wav_pred[0].to(torch.float32), cfg, "mel_train", "wav_train_target", "wav_train_output", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_wav(wav[0].to(torch.float32), wav_pred[0].to(torch.float32), cfg, "mel_train", "wav_train_target", "wav_train_output", current_time, ckpt_time)

    epoch_loss_disc /= iter_cnt
    epoch_loss_gen_stft /= iter_cnt
    epoch_loss_gen_gan /= iter_cnt
    epoch_loss_gen_all /= iter_cnt
    return epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all


def val_one_epoch_gan(
    gen,
    disc,
    val_loader,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    epoch_loss_disc = 0
    epoch_loss_gen_stft = 0
    epoch_loss_gen_gan = 0
    epoch_loss_gen_all = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("validation gan")
    gen.eval()
    disc.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
                wav_pred = gen(noise, feature)
                out_real = disc(wav)
                out_pred = disc(wav_pred)

                loss_disc = torch.mean((out_real - 1) ** 2) + torch.mean(out_pred ** 2)
                epoch_loss_disc += loss_disc.item()
                wandb.log({"val_loss_disc": loss_disc})

                loss_gen_stft = loss_f.calc_loss(wav, wav_pred)
                loss_gen_gan = torch.mean((out_pred - 1) ** 2)
                loss_gen_all =  cfg.train.stft_loss_weight * loss_gen_stft + cfg.train.gan_loss_weight * loss_gen_gan

                epoch_loss_gen_stft += loss_gen_stft.item()
                epoch_loss_gen_gan += loss_gen_gan.item()
                epoch_loss_gen_all += loss_gen_all.item()
                wandb.log({"val_loss_gen_stft": loss_gen_stft})
                wandb.log({"val_loss_gen_gan": loss_gen_gan})
                wandb.log({"val_loss_gen_all": loss_gen_all})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_wav(wav[0].to(torch.float32), wav_pred[0].to(torch.float32), cfg, "mel_validation", "wav_validation_target", "wav_validation_output", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_wav(wav[0].to(torch.float32), wav_pred[0].to(torch.float32), cfg, "mel_validation", "wav_validation_target", "wav_validation_output", current_time, ckpt_time)

    epoch_loss_disc /= iter_cnt
    epoch_loss_gen_stft /= iter_cnt
    epoch_loss_gen_gan /= iter_cnt
    epoch_loss_gen_all /= iter_cnt
    return epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
    fix_random_seed(cfg.train.random_seed)

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
    
    # train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_with_external_data(cfg, train_data_root, val_data_root)
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_pwg(cfg, train_data_root, val_data_root)

    loss_f = MultiResolutionSTFTLoss(
        n_fft_list=cfg.train.n_fft_list,
        hop_length_list=cfg.train.hop_length_list,
        win_length_list=cfg.train.win_length_list,
        device=device,
    )

    train_epoch_loss_disc_list = []
    train_epoch_loss_gen_stft_list = []
    train_epoch_loss_gen_gan_list = []
    train_epoch_loss_gen_all_list = []
    val_epoch_loss_disc_list = []
    val_epoch_loss_gen_stft_list = []
    val_epoch_loss_gen_gan_list = []
    val_epoch_loss_gen_all_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        gen, disc = make_model(cfg, device)

        if cfg.train.which_optim == 'adam':
            optimizer_g = torch.optim.Adam(
                params=gen.parameters(),
                lr=cfg.train.lr_gen,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )
            optimizer_d = torch.optim.Adam(
                params=disc.parameters(),
                lr=cfg.train.lr_disc,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )

        if cfg.train.which_scheduler == 'exp':
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_g, gamma=cfg.train.lr_decay_exp
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_d, gamma=cfg.train.lr_decay_exp
            )
        elif cfg.train.which_scheduler == 'warmup':
            scheduler_g = CosineLRScheduler(
                optimizer=optimizer_g,
                t_initial=cfg.train.max_epoch,
                lr_min=cfg.train.warmup_lr_min,
                warmup_t=int(cfg.train.max_epoch * cfg.train.warmup_t_rate),
                warmup_lr_init=cfg.train.warmup_lr_init,
                warmup_prefix=True,
            )
            scheduler_d = CosineLRScheduler(
                optimizer=optimizer_d,
                t_initial=cfg.train.max_epoch,
                lr_min=cfg.train.warmup_lr_min,
                warmup_t=int(cfg.train.max_epoch * cfg.train.warmup_t_rate),
                warmup_lr_init=cfg.train.warmup_lr_init,
                warmup_prefix=True,
            )

        scaler_g = torch.cuda.amp.GradScaler()
        scaler_d = torch.cuda.amp.GradScaler()

        last_epoch = 0
        if cfg.train.check_point_start:
            print("load check point")
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            gen.load_state_dict(checkpoint["gen"])
            disc.load_state_dict(checkpoint["disc"])
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            scheduler_d.load_state_dict(checkpoint["scheduler_d"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            train_epoch_loss_disc_list = checkpoint["train_epoch_loss_disc_list"]
            train_epoch_loss_gen_stft_list = checkpoint["train_epoch_loss_gen_stft_list"]
            train_epoch_loss_gen_gan_list = checkpoint["train_epoch_loss_gen_gan_list"]
            train_epoch_loss_gen_all_list = checkpoint["train_epoch_loss_gen_all_list"]
            val_epoch_loss_disc_list = checkpoint["val_epoch_loss_disc_list"]
            val_epoch_loss_gen_stft_list = checkpoint["val_epoch_loss_gen_stft_list"]
            val_epoch_loss_gen_gan_list = checkpoint["val_epoch_loss_gen_gan_list"]
            val_epoch_loss_gen_all_list = checkpoint["val_epoch_loss_gen_all_list"]
            last_epoch = checkpoint["epoch"]
        elif cfg.train.use_disc:
            if cfg.train.start_gan_training_pretrained_gen:
                print("load gen parameter")
                checkpoint_path = Path(cfg.train.gen_path).expanduser()
                if torch.cuda.is_available():
                    checkpoint = torch.load(checkpoint_path)
                else:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                gen.load_state_dict(checkpoint["gen"])

        if cfg.train.check_point_start_separate_save_dir:
            print("load check point (separate save dir)")
            checkpoint_path = Path(cfg.train.start_ckpt_path_separate_save_dir).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            gen.load_state_dict(checkpoint["gen"])
            disc.load_state_dict(checkpoint["disc"])

        wandb.watch(gen, **cfg.wandb_conf.watch)
        wandb.watch(disc, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            if cfg.train.use_disc:
                epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all = train_one_epoch_gan(
                    gen=gen,
                    disc=disc,
                    train_loader=train_loader,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                    loss_f=loss_f,
                    device=device,
                    cfg=cfg,
                    ckpt_time=ckpt_time,
                )
                train_epoch_loss_disc_list.append(epoch_loss_disc)
                train_epoch_loss_gen_stft_list.append(epoch_loss_gen_stft)
                train_epoch_loss_gen_gan_list.append(epoch_loss_gen_gan)
                train_epoch_loss_gen_all_list.append(epoch_loss_gen_all)

                epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all = val_one_epoch_gan(
                    gen=gen,
                    disc=disc,
                    val_loader=val_loader,
                    loss_f=loss_f,
                    device=device,
                    cfg=cfg,
                    ckpt_time=ckpt_time,
                )
                val_epoch_loss_disc_list.append(epoch_loss_disc)
                val_epoch_loss_gen_stft_list.append(epoch_loss_gen_stft)
                val_epoch_loss_gen_gan_list.append(epoch_loss_gen_gan)
                val_epoch_loss_gen_all_list.append(epoch_loss_gen_all)

                if cfg.train.which_scheduler == 'exp':
                    wandb.log({"learning_rate": scheduler_g.get_last_lr()[0]})
                    wandb.log({"learning_rate": scheduler_d.get_last_lr()[0]})
                    scheduler_g.step()
                    scheduler_d.step()
                elif cfg.train.which_scheduler == 'warmup':
                    wandb.log({"learning_rate": scheduler_g.optimizer.param_groups[0]['lr']})
                    wandb.log({"learning_rate": scheduler_d.optimizer.param_groups[0]['lr']})
                    scheduler_g.step(epoch)
                    scheduler_d.step(epoch)
            else:
                epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all = train_one_epoch(
                    gen=gen,
                    train_loader=train_loader,
                    optimizer_g=optimizer_g,
                    loss_f=loss_f,
                    device=device,
                    cfg=cfg,
                    ckpt_time=ckpt_time,
                )
                train_epoch_loss_disc_list.append(epoch_loss_disc)
                train_epoch_loss_gen_stft_list.append(epoch_loss_gen_stft)
                train_epoch_loss_gen_gan_list.append(epoch_loss_gen_gan)
                train_epoch_loss_gen_all_list.append(epoch_loss_gen_all)

                epoch_loss_disc, epoch_loss_gen_stft, epoch_loss_gen_gan, epoch_loss_gen_all = val_one_epoch(
                    gen=gen,
                    val_loader=val_loader,
                    loss_f=loss_f,
                    device=device,
                    cfg=cfg,
                    ckpt_time=ckpt_time,
                )
                val_epoch_loss_disc_list.append(epoch_loss_disc)
                val_epoch_loss_gen_stft_list.append(epoch_loss_gen_stft)
                val_epoch_loss_gen_gan_list.append(epoch_loss_gen_gan)
                val_epoch_loss_gen_all_list.append(epoch_loss_gen_all)

                if cfg.train.which_scheduler == 'exp':
                    wandb.log({"learning_rate": scheduler_g.get_last_lr()[0]})
                    scheduler_g.step()
                elif cfg.train.which_scheduler == 'warmup':
                    wandb.log({"learning_rate": scheduler_g.optimizer.param_groups[0]['lr']})
                    scheduler_g.step(epoch)

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    gen=gen,
                    disc=disc,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scheduler_g=scheduler_g,
                    scheduler_d=scheduler_d,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                    train_epoch_loss_disc_list=train_epoch_loss_disc_list,
                    train_epoch_loss_gen_stft_list=train_epoch_loss_gen_stft_list,
                    train_epoch_loss_gen_gan_list=train_epoch_loss_gen_gan_list,
                    train_epoch_loss_gen_all_list=train_epoch_loss_gen_all_list,
                    val_epoch_loss_disc_list=val_epoch_loss_disc_list,
                    val_epoch_loss_gen_stft_list=val_epoch_loss_gen_stft_list,
                    val_epoch_loss_gen_gan_list=val_epoch_loss_gen_gan_list,
                    val_epoch_loss_gen_all_list=val_epoch_loss_gen_all_list,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{current_epoch}.ckpt")
                )

            save_loss(train_epoch_loss_disc_list, val_epoch_loss_disc_list, save_path, "loss_disc")
            save_loss(train_epoch_loss_gen_stft_list, val_epoch_loss_gen_stft_list, save_path, "loss_gen_stft")
            save_loss(train_epoch_loss_gen_gan_list, val_epoch_loss_gen_gan_list, save_path, "loss_gen_gan")
            save_loss(train_epoch_loss_gen_all_list, val_epoch_loss_gen_all_list, save_path, "loss_gen_all")
            
    wandb.finish()

if __name__ == "__main__":
    main()