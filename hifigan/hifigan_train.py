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
import matplotlib.pyplot as plt
import random
from librosa.display import specshow
import itertools

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchaudio.transforms as AT

from utils import count_params, get_path_train, save_loss, check_mel_nar, make_train_val_loader, set_config, check_wav
from loss import MaskedLoss
from hifigan.model.generator import Generator
from hifigan.model.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from hifigan.hifigan_loss import feature_matching_loss, discriminator_loss, generator_loss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    gen, mpd, msd, optimizer_g, optimizer_d, scheduler_g, scheduler_d,
    train_epoch_disc_loss_mpd_list,
    train_epoch_disc_loss_msd_list,
    train_epoch_disc_loss_all_list,
    train_epoch_gen_loss_mel_list,
    train_epoch_gen_loss_fm_mpd_list,
    train_epoch_gen_loss_fm_msd_list,
    train_epoch_gen_loss_gan_mpd_list,
    train_epoch_gen_loss_gan_msd_list,
    train_epoch_gen_loss_all_list,
    val_epoch_disc_loss_mpd_list,
    val_epoch_disc_loss_msd_list,
    val_epoch_disc_loss_all_list,
    val_epoch_gen_loss_mel_list,
    val_epoch_gen_loss_fm_mpd_list,
    val_epoch_gen_loss_fm_msd_list,
    val_epoch_gen_loss_gan_mpd_list,
    val_epoch_gen_loss_gan_msd_list,
    val_epoch_gen_loss_all_list, 
    epoch, ckpt_path):
    torch.save({
        'gen': gen.state_dict(),
        'mpd': mpd.state_dict(),
        'msd': msd.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'scheduler_g': scheduler_g.state_dict(),
        'scheduler_d': scheduler_d.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'train_epoch_disc_loss_mpd_list': train_epoch_disc_loss_mpd_list,
        'train_epoch_disc_loss_msd_list': train_epoch_disc_loss_msd_list,
        'train_epoch_disc_loss_all_list': train_epoch_disc_loss_all_list,
        'train_epoch_gen_loss_mel_list': train_epoch_gen_loss_mel_list,
        'train_epoch_gen_loss_fm_mpd_list': train_epoch_gen_loss_fm_mpd_list,
        'train_epoch_gen_loss_fm_msd_list': train_epoch_gen_loss_fm_msd_list,
        'train_epoch_gen_loss_gan_mpd_list': train_epoch_gen_loss_gan_mpd_list,
        'train_epoch_gen_loss_gan_msd_list': train_epoch_gen_loss_gan_msd_list,
        'train_epoch_gen_loss_all_list': train_epoch_gen_loss_all_list,
        'val_epoch_disc_loss_mpd_list': val_epoch_disc_loss_mpd_list,
        'val_epoch_disc_loss_msd_list': val_epoch_disc_loss_msd_list,
        'val_epoch_disc_loss_all_list': val_epoch_disc_loss_all_list,
        'val_epoch_gen_loss_mel_list': val_epoch_gen_loss_mel_list,
        'val_epoch_gen_loss_fm_mpd_list': val_epoch_gen_loss_fm_mpd_list,
        'val_epoch_gen_loss_fm_msd_list': val_epoch_gen_loss_fm_msd_list,
        'val_epoch_gen_loss_gan_mpd_list': val_epoch_gen_loss_gan_mpd_list,
        'val_epoch_gen_loss_gan_msd_list': val_epoch_gen_loss_gan_msd_list,
        'val_epoch_gen_loss_all_list': val_epoch_gen_loss_all_list, 
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    gen = Generator(
        in_channels=cfg.model.in_channels,
        upsample_initial_channels=cfg.model.upsample_initial_channels,
        upsample_rates=cfg.model.upsample_rates,
        upsample_kernel_sizes=cfg.model.upsample_kernel_sizes,
        res_kernel_sizes=cfg.model.res_kernel_sizes,
        res_dilations=cfg.model.res_dilations,
    )
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    count_params(gen, "generator")
    count_params(mpd, "multi period discriminator")
    count_params(msd, "multi scale discriminator")

    return gen.to(device), mpd.to(device), msd.to(device)


def train_one_epoch(gen, mpd, msd, train_dataset, train_loader, optimizer_g, optimizer_d, loss_f, mel_transform, device, cfg, ckpt_time):
    epoch_disc_loss_mpd = 0
    epoch_disc_loss_msd = 0
    epoch_disc_loss_all = 0
    epoch_gen_loss_mel = 0
    epoch_gen_loss_fm_mpd = 0
    epoch_gen_loss_fm_msd = 0
    epoch_gen_loss_gan_mpd = 0
    epoch_gen_loss_gan_msd = 0
    epoch_gen_loss_all = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    gen.train()
    mpd.train()
    msd.train()

    feat_mean = train_dataset.feat_mean.to(device=device).unsqueeze(0).unsqueeze(-1)     # (1, C, 1)
    feat_std = train_dataset.feat_std.to(device=device).unsqueeze(0).unsqueeze(-1)       # (1, C, 1)

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, wav_q, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)
        data_len = data_len.to(device)

        wav_pred = gen(feature)

        mel_pred = mel_transform(wav_pred.squeeze(1))
        mel_pred = torch.log10(mel_pred)
        mel_pred = (mel_pred - feat_mean) / feat_std

        min_len = min(mel_pred.shape[-1], feature.shape[-1])
        mel_pred = mel_pred[..., :min_len]
        feature = feature[..., :min_len]

        out_real_list_mpd, out_pred_list_mpd, fmaps_real_list_mpd, fmaps_pred_list_mpd = mpd(wav, wav_pred.detach())
        out_real_list_msd, out_pred_list_msd, fmaps_real_list_msd, fmaps_pred_list_msd = msd(wav, wav_pred.detach())

        loss_mpd = discriminator_loss(out_real_list_mpd, out_pred_list_mpd)
        loss_msd = discriminator_loss(out_real_list_msd, out_pred_list_msd)
        loss_disc_all = loss_mpd + loss_msd

        epoch_disc_loss_mpd += loss_mpd.item()
        epoch_disc_loss_msd += loss_msd.item()
        epoch_disc_loss_all += loss_disc_all.item()
        wandb.log({"train_disc_loss_mpd": loss_mpd})
        wandb.log({"train_disc_loss_msd": loss_msd})
        wandb.log({"train_disc_loss_all": loss_disc_all})

        loss_disc_all.backward()
        optimizer_d.step()
        optimizer_d.zero_grad()

        out_real_list_mpd, out_pred_list_mpd, fmaps_real_list_mpd, fmaps_pred_list_mpd = mpd(wav, wav_pred)
        out_real_list_msd, out_pred_list_msd, fmaps_real_list_msd, fmaps_pred_list_msd = msd(wav, wav_pred)

        loss_mel = loss_f.l1_loss(mel_pred, feature, data_len, wav_pred.shape[-1] // cfg.model.hop_length)
        loss_fm_mpd = feature_matching_loss(fmaps_real_list_mpd, fmaps_pred_list_mpd)
        loss_fm_msd = feature_matching_loss(fmaps_real_list_msd, fmaps_pred_list_msd)
        loss_gan_mpd = generator_loss(out_pred_list_mpd)
        loss_gan_msd = generator_loss(out_pred_list_msd)
        loss_gen_all = loss_mel * cfg.train.mel_loss_weight + (loss_fm_mpd + loss_fm_msd) * cfg.train.fm_loss_weight +\
             (loss_gan_mpd + loss_gan_msd) * cfg.train.gan_loss_weight

        epoch_gen_loss_mel += loss_mel.item()
        epoch_gen_loss_fm_mpd += loss_fm_mpd.item()
        epoch_gen_loss_fm_msd += loss_fm_msd.item()
        epoch_gen_loss_gan_mpd += loss_gan_mpd.item()
        epoch_gen_loss_gan_msd += loss_gan_msd.item()
        epoch_gen_loss_all += loss_gen_all.item()
        wandb.log({"train_gen_loss_mel": loss_mel})
        wandb.log({"train_gen_loss_fm_mpd": loss_fm_mpd})
        wandb.log({"train_gen_loss_fm_msd": loss_fm_msd})
        wandb.log({"train_gen_loss_gan_mpd": loss_gan_mpd})
        wandb.log({"train_gen_loss_gan_msd": loss_gan_msd})
        wandb.log({"train_gen_loss_all": loss_gen_all})

        loss_gen_all.backward()
        optimizer_g.step()
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

    epoch_disc_loss_mpd /= iter_cnt
    epoch_disc_loss_msd /= iter_cnt
    epoch_disc_loss_all /= iter_cnt
    epoch_gen_loss_mel /= iter_cnt
    epoch_gen_loss_fm_mpd /= iter_cnt
    epoch_gen_loss_fm_msd /= iter_cnt
    epoch_gen_loss_gan_mpd /= iter_cnt
    epoch_gen_loss_gan_msd /= iter_cnt
    epoch_gen_loss_all /= iter_cnt
    return (
        epoch_disc_loss_mpd, epoch_disc_loss_msd, epoch_disc_loss_all, 
        epoch_gen_loss_mel, epoch_gen_loss_fm_mpd, epoch_gen_loss_fm_msd, 
        epoch_gen_loss_gan_mpd, epoch_gen_loss_gan_msd, epoch_gen_loss_all
    )


def val_one_epoch(gen, mpd, msd, val_dataset, val_loader, loss_f, mel_transform, device, cfg, ckpt_time):
    epoch_disc_loss_mpd = 0
    epoch_disc_loss_msd = 0
    epoch_disc_loss_all = 0
    epoch_gen_loss_mel = 0
    epoch_gen_loss_fm_mpd = 0
    epoch_gen_loss_fm_msd = 0
    epoch_gen_loss_gan_mpd = 0
    epoch_gen_loss_gan_msd = 0
    epoch_gen_loss_all = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start") 
    gen.eval()
    mpd.eval()
    msd.eval()

    feat_mean = val_dataset.feat_mean.to(device=device).unsqueeze(0).unsqueeze(-1)     # (1, C, 1)
    feat_std = val_dataset.feat_std.to(device=device).unsqueeze(0).unsqueeze(-1)       # (1, C, 1)

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, wav_q, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)
        data_len = data_len.to(device)

        with torch.no_grad():
            wav_pred = gen(feature)

        mel_pred = mel_transform(wav_pred.squeeze(1))
        mel_pred = torch.log10(mel_pred)
        mel_pred = (mel_pred - feat_mean) / feat_std

        min_len = min(mel_pred.shape[-1], feature.shape[-1])
        mel_pred = mel_pred[..., :min_len]
        feature = feature[..., :min_len]

        with torch.no_grad():
            out_real_list_mpd, out_pred_list_mpd, fmaps_real_list_mpd, fmaps_pred_list_mpd = mpd(wav, wav_pred.detach())
            out_real_list_msd, out_pred_list_msd, fmaps_real_list_msd, fmaps_pred_list_msd = msd(wav, wav_pred.detach())

        loss_mpd = discriminator_loss(out_real_list_mpd, out_pred_list_mpd)
        loss_msd = discriminator_loss(out_real_list_msd, out_pred_list_msd)
        loss_disc_all = loss_mpd + loss_msd

        epoch_disc_loss_mpd += loss_mpd.item()
        epoch_disc_loss_msd += loss_msd.item()
        epoch_disc_loss_all += loss_disc_all.item()
        wandb.log({"val_disc_loss_mpd": loss_mpd})
        wandb.log({"val_disc_loss_msd": loss_msd})
        wandb.log({"val_disc_loss_all": loss_disc_all})

        with torch.no_grad():
            out_real_list_mpd, out_pred_list_mpd, fmaps_real_list_mpd, fmaps_pred_list_mpd = mpd(wav, wav_pred)
            out_real_list_msd, out_pred_list_msd, fmaps_real_list_msd, fmaps_pred_list_msd = msd(wav, wav_pred)

        loss_mel = loss_f.l1_loss(mel_pred, feature, data_len, wav_pred.shape[-1] // cfg.model.hop_length)
        loss_fm_mpd = feature_matching_loss(fmaps_real_list_mpd, fmaps_pred_list_mpd)
        loss_fm_msd = feature_matching_loss(fmaps_real_list_msd, fmaps_pred_list_msd)
        loss_gan_mpd = generator_loss(out_pred_list_mpd)
        loss_gan_msd = generator_loss(out_pred_list_msd)
        loss_gen_all = loss_mel * cfg.train.mel_loss_weight + (loss_fm_mpd + loss_fm_msd) * cfg.train.fm_loss_weight +\
             (loss_gan_mpd + loss_gan_msd) * cfg.train.gan_loss_weight

        epoch_gen_loss_mel += loss_mel.item()
        epoch_gen_loss_fm_mpd += loss_fm_mpd.item()
        epoch_gen_loss_fm_msd += loss_fm_msd.item()
        epoch_gen_loss_gan_mpd += loss_gan_mpd.item()
        epoch_gen_loss_gan_msd += loss_gan_msd.item()
        epoch_gen_loss_all += loss_gen_all.item()
        wandb.log({"val_gen_loss_mel": loss_mel})
        wandb.log({"val_gen_loss_fm_mpd": loss_fm_mpd})
        wandb.log({"val_gen_loss_fm_msd": loss_fm_msd})
        wandb.log({"val_gen_loss_gan_mpd": loss_gan_mpd})
        wandb.log({"val_gen_loss_gan_msd": loss_gan_msd})
        wandb.log({"val_gen_loss_all": loss_gen_all})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_wav(wav[0], wav_pred[0], cfg, "mel_validation", "wav_validation_target", "wav_validation_output", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_wav(wav[0], wav_pred[0], cfg, "mel_validation", "wav_validation_target", "wav_validation_output", current_time, ckpt_time)

    epoch_disc_loss_mpd /= iter_cnt
    epoch_disc_loss_msd /= iter_cnt
    epoch_gen_loss_mel /= iter_cnt
    epoch_gen_loss_fm_mpd /= iter_cnt
    epoch_gen_loss_fm_msd /= iter_cnt
    epoch_gen_loss_gan_mpd /= iter_cnt
    epoch_gen_loss_gan_msd /= iter_cnt
    return (
        epoch_disc_loss_mpd, epoch_disc_loss_msd, epoch_disc_loss_all, 
        epoch_gen_loss_mel, epoch_gen_loss_fm_mpd, epoch_gen_loss_fm_msd, 
        epoch_gen_loss_gan_mpd, epoch_gen_loss_gan_msd, epoch_gen_loss_all
    )


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

    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader(cfg, train_data_root, val_data_root)

    loss_f = MaskedLoss()

    mel_transform = AT.MelSpectrogram(
        sample_rate=cfg.model.sampling_rate,
        n_fft=cfg.model.n_fft,
        win_length=cfg.model.win_length,
        hop_length=cfg.model.hop_length,
        f_min=cfg.model.f_min,
        f_max=cfg.model.f_max,
        n_mels=cfg.model.n_mel_channels,
    ).to(device)

    train_epoch_disc_loss_mpd_list = []
    train_epoch_disc_loss_msd_list = []
    train_epoch_disc_loss_all_list = []
    train_epoch_gen_loss_mel_list = []
    train_epoch_gen_loss_fm_mpd_list = []
    train_epoch_gen_loss_fm_msd_list = []
    train_epoch_gen_loss_gan_mpd_list = []
    train_epoch_gen_loss_gan_msd_list = []
    train_epoch_gen_loss_all_list = []
    val_epoch_disc_loss_mpd_list = []
    val_epoch_disc_loss_msd_list = []
    val_epoch_disc_loss_all_list = []
    val_epoch_gen_loss_mel_list = []
    val_epoch_gen_loss_fm_mpd_list = []
    val_epoch_gen_loss_fm_msd_list = []
    val_epoch_gen_loss_gan_mpd_list = []
    val_epoch_gen_loss_gan_msd_list = []
    val_epoch_gen_loss_all_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        gen, mpd, msd = make_model(cfg, device)

        optimizer_g = torch.optim.Adam(
            params=gen.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )
        optimizer_d = torch.optim.Adam(
            params=itertools.chain(mpd.parameters(), msd.parameters()),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g, gamma=cfg.train.lr_decay
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d, gamma=cfg.train.lr_decay
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            print("load check point")
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            gen.load_state_dict(checkpoint["gen"])
            mpd.load_state_dict(checkpoint["mpd"])
            msd.load_state_dict(checkpoint["msd"])
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            scheduler_d.load_state_dict(checkpoint["scheduler_d"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_epoch_disc_loss_mpd_list = checkpoint["train_epoch_disc_loss_mpd_list"]
            train_epoch_disc_loss_msd_list = checkpoint["train_epoch_disc_loss_msd_list"]
            train_epoch_disc_loss_all_list = checkpoint["train_epoch_disc_loss_all_list"]
            train_epoch_gen_loss_mel_list = checkpoint["train_epoch_gen_loss_mel_list"]
            train_epoch_gen_loss_fm_mpd_list = checkpoint["train_epoch_gen_loss_fm_mpd_list"]
            train_epoch_gen_loss_fm_msd_list = checkpoint["train_epoch_gen_loss_fm_msd_list"]
            train_epoch_gen_loss_gan_mpd_list = checkpoint["train_epoch_gen_loss_gan_mpd_list"]
            train_epoch_gen_loss_gan_msd_list = checkpoint["train_epoch_gen_loss_gan_msd_list"]
            train_epoch_gen_loss_all_list = checkpoint["train_epoch_gen_loss_all_list"]
            val_epoch_disc_loss_mpd_list = checkpoint["val_epoch_disc_loss_mpd_list"]
            val_epoch_disc_loss_msd_list = checkpoint["val_epoch_disc_loss_msd_list"]
            val_epoch_disc_loss_all_list = checkpoint["val_epoch_disc_loss_all_list"]
            val_epoch_gen_loss_mel_list = checkpoint["val_epoch_gen_loss_mel_list"]
            val_epoch_gen_loss_fm_mpd_list = checkpoint["val_epoch_gen_loss_fm_mpd_list"]
            val_epoch_gen_loss_fm_msd_list = checkpoint["val_epoch_gen_loss_fm_msd_list"]
            val_epoch_gen_loss_gan_mpd_list = checkpoint["val_epoch_gen_loss_gan_mpd_list"]
            val_epoch_gen_loss_gan_msd_list = checkpoint["val_epoch_gen_loss_gan_msd_list"]
            val_epoch_gen_loss_all_list = checkpoint["val_epoch_gen_loss_all_list"]
        
        wandb.watch(gen, **cfg.wandb_conf.watch)
        wandb.watch(mpd, **cfg.wandb_conf.watch)
        wandb.watch(msd, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            epoch_disc_loss_mpd, epoch_disc_loss_msd, epoch_disc_loss_all, epoch_gen_loss_mel,\
                 epoch_gen_loss_fm_mpd, epoch_gen_loss_fm_msd, epoch_gen_loss_gan_mpd, epoch_gen_loss_gan_msd, epoch_gen_loss_all = train_one_epoch(
                gen=gen,
                mpd=mpd,
                msd=msd,
                train_dataset=train_dataset,
                train_loader=train_loader,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                loss_f=loss_f,
                mel_transform=mel_transform,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            train_epoch_disc_loss_mpd_list.append(epoch_disc_loss_mpd)
            train_epoch_disc_loss_msd_list.append(epoch_disc_loss_msd)
            train_epoch_disc_loss_all_list.append(epoch_disc_loss_all)
            train_epoch_gen_loss_mel_list.append(epoch_gen_loss_mel)
            train_epoch_gen_loss_fm_mpd_list.append(epoch_gen_loss_fm_mpd)
            train_epoch_gen_loss_fm_msd_list.append(epoch_gen_loss_fm_msd)
            train_epoch_gen_loss_gan_mpd_list.append(epoch_gen_loss_gan_mpd)
            train_epoch_gen_loss_gan_msd_list.append(epoch_gen_loss_gan_msd)
            train_epoch_gen_loss_all_list.append(epoch_gen_loss_all)

            epoch_disc_loss_mpd, epoch_disc_loss_msd, epoch_disc_loss_all, epoch_gen_loss_mel,\
                 epoch_gen_loss_fm_mpd, epoch_gen_loss_fm_msd, epoch_gen_loss_gan_mpd, epoch_gen_loss_gan_msd, epoch_gen_loss_all = val_one_epoch(
                gen=gen,
                mpd=mpd,
                msd=msd,
                val_dataset=val_dataset,
                val_loader=val_loader,
                loss_f=loss_f,
                mel_transform=mel_transform,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_epoch_disc_loss_mpd_list.append(epoch_disc_loss_mpd)
            val_epoch_disc_loss_msd_list.append(epoch_disc_loss_msd)
            val_epoch_disc_loss_all_list.append(epoch_disc_loss_all)
            val_epoch_gen_loss_mel_list.append(epoch_gen_loss_mel)
            val_epoch_gen_loss_fm_mpd_list.append(epoch_gen_loss_fm_mpd)
            val_epoch_gen_loss_fm_msd_list.append(epoch_gen_loss_fm_msd)
            val_epoch_gen_loss_gan_mpd_list.append(epoch_gen_loss_gan_mpd)
            val_epoch_gen_loss_gan_msd_list.append(epoch_gen_loss_gan_msd)
            val_epoch_gen_loss_all_list.append(epoch_gen_loss_all)

            scheduler_d.step()
            scheduler_g.step()

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    gen=gen,
                    mpd=mpd,
                    msd=msd,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scheduler_g=scheduler_g,
                    scheduler_d=scheduler_d,
                    train_epoch_disc_loss_mpd_list=train_epoch_disc_loss_mpd_list,
                    train_epoch_disc_loss_msd_list=train_epoch_disc_loss_msd_list,
                    train_epoch_disc_loss_all_list=train_epoch_disc_loss_all_list,
                    train_epoch_gen_loss_mel_list=train_epoch_gen_loss_mel_list,
                    train_epoch_gen_loss_fm_mpd_list=train_epoch_gen_loss_fm_mpd_list,
                    train_epoch_gen_loss_fm_msd_list=train_epoch_gen_loss_fm_msd_list,
                    train_epoch_gen_loss_gan_mpd_list=train_epoch_gen_loss_gan_mpd_list,
                    train_epoch_gen_loss_gan_msd_list=train_epoch_gen_loss_gan_msd_list,
                    train_epoch_gen_loss_all_list=train_epoch_gen_loss_all_list,
                    val_epoch_disc_loss_mpd_list=val_epoch_disc_loss_mpd_list,
                    val_epoch_disc_loss_msd_list=val_epoch_disc_loss_msd_list,
                    val_epoch_disc_loss_all_list=val_epoch_disc_loss_all_list,
                    val_epoch_gen_loss_mel_list=val_epoch_gen_loss_mel_list,
                    val_epoch_gen_loss_fm_mpd_list=val_epoch_gen_loss_fm_mpd_list,
                    val_epoch_gen_loss_fm_msd_list=val_epoch_gen_loss_fm_msd_list,
                    val_epoch_gen_loss_gan_mpd_list=val_epoch_gen_loss_gan_mpd_list,
                    val_epoch_gen_loss_gan_msd_list=val_epoch_gen_loss_gan_msd_list,
                    val_epoch_gen_loss_all_list=val_epoch_gen_loss_all_list, 
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt"),
                )

            save_loss(train_epoch_disc_loss_mpd_list, val_epoch_disc_loss_mpd_list, save_path, "disc_loss_mpd")
            save_loss(train_epoch_disc_loss_msd_list, val_epoch_disc_loss_msd_list, save_path, "disc_loss_msd")
            save_loss(train_epoch_disc_loss_all_list, val_epoch_disc_loss_all_list, save_path, "disc_loss_all")
            save_loss(train_epoch_gen_loss_mel_list, val_epoch_gen_loss_mel_list, save_path, "gen_loss_mel")
            save_loss(train_epoch_gen_loss_fm_mpd_list, val_epoch_gen_loss_fm_mpd_list, save_path, "gen_loss_fm_mpd")
            save_loss(train_epoch_gen_loss_fm_msd_list, val_epoch_gen_loss_fm_msd_list, save_path, "gen_loss_fm_msd")
            save_loss(train_epoch_gen_loss_gan_mpd_list, val_epoch_gen_loss_gan_mpd_list, save_path, "gen_loss_gan_mpd")
            save_loss(train_epoch_gen_loss_gan_msd_list, val_epoch_gen_loss_gan_msd_list, save_path, "gen_loss_gan_msd")
            save_loss(train_epoch_gen_loss_all_list, val_epoch_gen_loss_all_list, save_path, "gen_loss_all")

        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(gen.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__ == "__main__":
    main()