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
from torch.nn.utils import clip_grad_norm_

from utils import set_config, get_path_train, make_train_val_loader, count_params, save_loss, check_movie, requires_grad_change
from model.face_gen import Generator, FrameDiscriminator, MultipleFrameDiscriminator, SequenceDiscriminator, SyncDiscriminator
from model.transformer_remake import make_pad_mask

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    gen, frame_disc, seq_disc, sync_disc,
    optimizer_gen, optimizer_frame_disc, optimizer_multiframe_disc, optimizer_seq_disc, optimizer_sync_disc,
    scheduler_gen, scheduler_frame_disc, scheduler_multiframe_disc, scheduler_seq_disc, scheduler_sync_disc,
    train_epoch_frame_disc_loss_list,
    train_epoch_multiframe_disc_loss_list,
    train_epoch_seq_disc_loss_list,
    train_epoch_sync_disc_loss_list,
    train_epoch_gen_l1_loss_list,
    train_epoch_gen_frame_disc_loss_list,
    train_epoch_gen_multiframe_disc_loss_list,
    train_epoch_gen_seq_disc_loss_list,
    train_epoch_gen_sync_disc_loss_list,
    train_epoch_gen_loss_all_list,
    val_epoch_frame_disc_loss_list,
    val_epoch_multiframe_disc_loss_list,
    val_epoch_seq_disc_loss_list,
    val_epoch_sync_disc_loss_list,
    val_epoch_gen_l1_loss_list,
    val_epoch_gen_frame_disc_loss_list,
    val_epoch_gen_multiframe_disc_loss_list,
    val_epoch_gen_seq_disc_loss_list,
    val_epoch_gen_sync_disc_loss_list,
    val_epoch_gen_loss_all_list,
    epoch, ckpt_path):
    torch.save({
        'gen': gen.state_dict(),
        'frame_disc': frame_disc.state_dict(),
        'seq_disc': seq_disc.state_dict(),
        'sync_disc': sync_disc.state_dict(),
        'optimizer_gen': optimizer_gen.state_dict(),
        'optimizer_frame_disc': optimizer_frame_disc.state_dict(),
        'optimizer_multiframe_disc': optimizer_multiframe_disc.state_dict(),
        'optimizer_seq_disc': optimizer_seq_disc.state_dict(),
        'optimizer_sync_disc': optimizer_sync_disc.state_dict(),
        'scheduler_gen': scheduler_gen.state_dict(),
        'scheduler_frame_disc': scheduler_frame_disc.state_dict(),
        'scheduler_multiframe_disc': scheduler_multiframe_disc.state_dict(),
        'scheduler_seq_disc': scheduler_seq_disc.state_dict(),
        'scheduler_sync_disc': scheduler_sync_disc.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        "train_epoch_frame_disc_loss_list" : train_epoch_frame_disc_loss_list,
        "train_epoch_multiframe_disc_loss_list" : train_epoch_multiframe_disc_loss_list,
        "train_epoch_seq_disc_loss_list" : train_epoch_seq_disc_loss_list,
        "train_epoch_sync_disc_loss_list" : train_epoch_sync_disc_loss_list,
        "train_epoch_gen_l1_loss_list" : train_epoch_gen_l1_loss_list,
        "train_epoch_gen_frame_disc_loss_list" : train_epoch_gen_frame_disc_loss_list,
        "train_epoch_gen_multiframe_disc_loss_list" : train_epoch_gen_multiframe_disc_loss_list,
        "train_epoch_gen_seq_disc_loss_list" : train_epoch_gen_seq_disc_loss_list,
        "train_epoch_gen_sync_disc_loss_list" : train_epoch_gen_sync_disc_loss_list,
        "train_epoch_gen_loss_all_list" : train_epoch_gen_loss_all_list,
        "val_epoch_frame_disc_loss_list" : val_epoch_frame_disc_loss_list,
        "val_epoch_multiframe_disc_loss_list" : val_epoch_multiframe_disc_loss_list,
        "val_epoch_seq_disc_loss_list" : val_epoch_seq_disc_loss_list,
        "val_epoch_sync_disc_loss_list" : val_epoch_sync_disc_loss_list,
        "val_epoch_gen_l1_loss_list" : val_epoch_gen_l1_loss_list,
        "val_epoch_gen_frame_disc_loss_list" : val_epoch_gen_frame_disc_loss_list,
        "val_epoch_gen_multiframe_disc_loss_list" : val_epoch_gen_multiframe_disc_loss_list,
        "val_epoch_gen_seq_disc_loss_list" : val_epoch_gen_seq_disc_loss_list,
        "val_epoch_gen_sync_disc_loss_list" : val_epoch_gen_sync_disc_loss_list,
        "val_epoch_gen_loss_all_list" : val_epoch_gen_loss_all_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    gen = Generator(
        in_channels=cfg.model.in_channels,
        img_hidden_channels=cfg.model.face_gen_img_hidden_channels,
        img_cond_channels=cfg.model.face_gen_img_cond_channels,
        feat_channels=cfg.model.n_mel_channels,
        feat_cond_channels=cfg.model.face_gen_feat_cond_channels,
        mel_enc_hidden_channels=cfg.model.face_gen_mel_enc_hidden_channels,
        noise_channels=cfg.model.face_gen_noise_channels,
        tc_ksize=cfg.model.face_gen_tc_ksize,
        dropout=cfg.train.gen_dropout,
        is_large=cfg.model.is_large,
        fps=cfg.model.fps,
    )
    frame_disc = FrameDiscriminator(
        in_channels=int(cfg.model.in_channels * 2), 
        dropout=cfg.train.disc_dropout,
    )
    multiframe_disc = MultipleFrameDiscriminator(
        in_channels=cfg.model.in_channels,
        dropout=cfg.train.disc_dropout,
    )
    seq_disc = SequenceDiscriminator(
        in_channels=cfg.model.in_channels,
        feat_channels=cfg.model.n_mel_channels,
        dropout=cfg.train.disc_dropout,
        fps=cfg.model.fps,
    )
    sync_disc = SyncDiscriminator(
        in_channels=cfg.model.in_channels,
        feat_channels=cfg.model.n_mel_channels,
        dropout=cfg.train.disc_dropout,
        fps=cfg.model.fps,
    )
    count_params(gen, "gen")
    count_params(frame_disc, "frame_disc")
    count_params(multiframe_disc, "multiframe_disc")
    count_params(seq_disc, "seq_disc")
    count_params(sync_disc, "sync_disc")
    return gen.to(device), frame_disc.to(device), multiframe_disc.to(device), seq_disc.to(device), sync_disc.to(device)


def train_one_epoch(
    gen, frame_disc, multiframe_disc, seq_disc, sync_disc, train_loader, train_dataset, 
    optimizer_gen, optimizer_frame_disc, optimizer_multiframe_disc, optimizer_seq_disc, optimizer_sync_disc, device, cfg, ckpt_time):
    epoch_frame_disc_loss = 0
    epoch_multiframe_disc_loss = 0
    epoch_seq_disc_loss = 0
    epoch_sync_disc_loss = 0
    epoch_gen_l1_loss = 0
    epoch_gen_frame_disc_loss = 0
    epoch_gen_multiframe_disc_loss = 0
    epoch_gen_seq_disc_loss = 0
    epoch_gen_sync_disc_loss = 0
    epoch_gen_loss_all = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    gen.train()
    frame_disc.train()
    multiframe_disc.train()
    seq_disc.train()
    sync_disc.train()
    r = cfg.model.reduction_factor
    lip_mean = train_dataset.lip_mean
    lip_std = train_dataset.lip_std

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        lip_len = torch.clamp(lip_len, max=lip.shape[-1])

        output = gen(lip[..., 0], feature, lip_len)

        ### optimize disc ###
        frame_disc = requires_grad_change(frame_disc, True)
        multiframe_disc = requires_grad_change(multiframe_disc, True)
        seq_disc = requires_grad_change(seq_disc, True)
        sync_disc = requires_grad_change(sync_disc, True)
        
        # frame disc
        frame_index = random.randint(0, torch.min(lip_len).item() - 1)
        fake_frame = frame_disc(output.detach()[..., frame_index].squeeze(-1), lip[..., 0])
        real_frame = frame_disc(lip[..., frame_index].squeeze(-1), lip[..., 0])
        frame_disc_loss = F.mse_loss(fake_frame, torch.zeros_like(fake_frame)) + F.mse_loss(real_frame, torch.ones_like(real_frame))

        # multiframe disc
        multiframe_start_index = random.randint(0, torch.min(lip_len).item() - int(cfg.train.multiframe_disc_input_sec * cfg.model.fps) - 1)
        fake_multiframe = multiframe_disc(output.detach()[..., multiframe_start_index:multiframe_start_index + int(cfg.train.multiframe_disc_input_sec * cfg.model.fps)])
        real_multiframe = multiframe_disc(lip[..., multiframe_start_index:multiframe_start_index + int(cfg.train.multiframe_disc_input_sec * cfg.model.fps)])
        multiframe_disc_loss = F.mse_loss(fake_multiframe, torch.zeros_like(fake_multiframe)) + F.mse_loss(real_multiframe, torch.ones_like(real_multiframe))

        # seq disc
        if torch.min(lip_len).item() - 1 - int(cfg.train.seq_disc_input_sec * cfg.model.fps) < 0:
            seq_start_index = 0
        else:
            seq_start_index = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.seq_disc_input_sec * cfg.model.fps))
        fake_seq = seq_disc(
            output.detach()[..., seq_start_index:seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)], 
            feature[..., int(seq_start_index * r):int((seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)) * r)]
        )
        real_seq = seq_disc(
            lip[..., seq_start_index:seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)], 
            feature[..., int(seq_start_index * r):int((seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)) * r)]
        )
        seq_disc_loss = F.mse_loss(fake_seq, torch.zeros_like(fake_seq)) + F.mse_loss(real_seq, torch.ones_like(real_seq))

        # sync disc
        sync_start_index = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))
        sync_start_index_fake = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))
        while sync_start_index == sync_start_index_fake:
            sync_start_index_fake = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))

        lip_crop = lip[..., sync_start_index:sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)]
        output_crop = output.detach()[..., sync_start_index:sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)]
        feature_crop_real = feature[..., int(sync_start_index * r):int((sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)) * r)]
        feature_crop_fake = feature[..., int(sync_start_index_fake * r):int((sync_start_index_fake + int(cfg.train.sync_disc_input_sec * cfg.model.fps)) * r)]

        fake_sync = sync_disc(output_crop, feature_crop_real)
        not_sync = sync_disc(lip_crop, feature_crop_fake)
        real_sync = sync_disc(lip_crop, feature_crop_real)
        sync_disc_loss = F.mse_loss(fake_sync, torch.zeros_like(fake_sync)) + F.mse_loss(not_sync, torch.zeros_like(not_sync))\
            + F.mse_loss(real_sync, torch.ones_like(real_sync))

        frame_disc_loss.backward()
        multiframe_disc_loss.backward()
        seq_disc_loss.backward()
        sync_disc_loss.backward()
        optimizer_frame_disc.step()
        optimizer_multiframe_disc.step()
        optimizer_seq_disc.step()
        optimizer_sync_disc.step()
        optimizer_frame_disc.zero_grad()
        optimizer_multiframe_disc.zero_grad()
        optimizer_seq_disc.zero_grad()
        optimizer_sync_disc.zero_grad()
        optimizer_gen.zero_grad()

        epoch_frame_disc_loss += frame_disc_loss.item()
        epoch_multiframe_disc_loss += multiframe_disc_loss.item()
        epoch_seq_disc_loss += seq_disc_loss.item()
        epoch_sync_disc_loss += sync_disc_loss.item()
        wandb.log({"train_frame_disc_loss": frame_disc_loss})
        wandb.log({"train_multiframe_disc_loss": multiframe_disc_loss})
        wandb.log({"train_seq_disc_loss": seq_disc_loss})
        wandb.log({"train_sync_disc_loss": sync_disc_loss})

        ### optimize generator ###
        frame_disc = requires_grad_change(frame_disc, False)
        multiframe_disc = requires_grad_change(multiframe_disc, False)
        seq_disc = requires_grad_change(seq_disc, False)
        sync_disc = requires_grad_change(sync_disc, False)
        
        # frame disc
        frame_index = random.randint(0, torch.min(lip_len).item() - 1)
        fake_frame = frame_disc(output[..., frame_index].squeeze(-1), lip[..., 0])

        # multiframe disc
        multiframe_start_index = random.randint(0, torch.min(lip_len).item() - int(cfg.train.multiframe_disc_input_sec * cfg.model.fps) - 1)
        fake_multiframe = multiframe_disc(output[..., multiframe_start_index:multiframe_start_index + int(cfg.train.multiframe_disc_input_sec * cfg.model.fps)])

        # seq disc
        if torch.min(lip_len).item() - 1 - int(cfg.train.seq_disc_input_sec * cfg.model.fps) < 0:
            seq_start_index = 0
        else:
            seq_start_index = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.seq_disc_input_sec * cfg.model.fps))
        fake_seq = seq_disc(
            output[..., seq_start_index:seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)], 
            feature[..., int(seq_start_index * r):int((seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)) * r)]
        )

        # sync disc
        sync_start_index = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))
        output_crop = output[..., sync_start_index:sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)]
        feature_crop_real = feature[..., int(sync_start_index * r):int((sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)) * r)]
        fake_sync = sync_disc(output_crop, feature_crop_real)

        # masked selectする前に一旦可視化
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_movie(lip[0], output[0].detach(), lip_mean, lip_std, cfg, "train_movie", current_time, ckpt_time)
                break
        if iter_cnt % (all_iter - 1) == 0:
            check_movie(lip[0], output[0].detach(), lip_mean, lip_std, cfg, "train_movie", current_time, ckpt_time)

        mask = 1.0 - make_pad_mask(lip_len, lip.shape[-1]).to(torch.float32).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 1, T)
        mask = mask.to(torch.bool)
        output = torch.masked_select(output, mask)
        lip = torch.masked_select(lip, mask)
        gen_l1_loss = F.l1_loss(output, lip)

        gen_frame_disc_loss = F.mse_loss(fake_frame, torch.ones_like(fake_frame))
        gen_multiframe_disc_loss = F.mse_loss(fake_multiframe, torch.ones_like(fake_multiframe))
        gen_seq_disc_loss = F.mse_loss(fake_seq, torch.ones_like(fake_seq))
        gen_sync_disc_loss = F.mse_loss(fake_sync, torch.ones_like(fake_sync))
        gen_loss_all = cfg.train.l1_weight * gen_l1_loss + cfg.train.frame_disc_weight * gen_frame_disc_loss \
            + cfg.train.multiframe_disc_weight * gen_multiframe_disc_loss + cfg.train.seq_disc_weight * gen_seq_disc_loss \
                + cfg.train.sync_disc_weight * gen_sync_disc_loss

        gen_loss_all.backward()
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        optimizer_frame_disc.zero_grad()
        optimizer_multiframe_disc.zero_grad()
        optimizer_seq_disc.zero_grad()
        optimizer_sync_disc.zero_grad()

        epoch_gen_l1_loss += gen_l1_loss.item()
        epoch_gen_frame_disc_loss += gen_frame_disc_loss.item()
        epoch_gen_multiframe_disc_loss += gen_multiframe_disc_loss.item()
        epoch_gen_seq_disc_loss += gen_seq_disc_loss.item()
        epoch_gen_sync_disc_loss += gen_sync_disc_loss.item()
        epoch_gen_loss_all += gen_loss_all.item()
        wandb.log({"train_gen_l1_loss": gen_l1_loss})
        wandb.log({"train_gen_frame_disc_loss": gen_frame_disc_loss})
        wandb.log({"train_gen_multiframe_disc_loss": gen_multiframe_disc_loss})
        wandb.log({"train_gen_seq_disc_loss": gen_seq_disc_loss})
        wandb.log({"train_gen_sync_disc_loss": gen_sync_disc_loss})
        wandb.log({"train_gen_loss_all": gen_loss_all})

    epoch_frame_disc_loss /= iter_cnt
    epoch_multiframe_disc_loss /= iter_cnt
    epoch_seq_disc_loss /= iter_cnt
    epoch_sync_disc_loss /= iter_cnt
    epoch_gen_l1_loss /= iter_cnt
    epoch_gen_frame_disc_loss /= iter_cnt
    epoch_gen_multiframe_disc_loss /= iter_cnt
    epoch_gen_seq_disc_loss /= iter_cnt
    epoch_gen_sync_disc_loss /= iter_cnt
    epoch_gen_loss_all /= iter_cnt
    return epoch_frame_disc_loss, epoch_multiframe_disc_loss,  epoch_seq_disc_loss, epoch_sync_disc_loss, \
        epoch_gen_l1_loss, epoch_gen_frame_disc_loss, epoch_gen_multiframe_disc_loss, \
            epoch_gen_seq_disc_loss, epoch_gen_sync_disc_loss, epoch_gen_loss_all


def val_one_epoch(
    gen, frame_disc, multiframe_disc, seq_disc, sync_disc, val_loader, val_dataset, device, cfg, ckpt_time):
    epoch_frame_disc_loss = 0
    epoch_multiframe_disc_loss = 0
    epoch_seq_disc_loss = 0
    epoch_sync_disc_loss = 0
    epoch_gen_l1_loss = 0
    epoch_gen_frame_disc_loss = 0
    epoch_gen_multiframe_disc_loss = 0
    epoch_gen_seq_disc_loss = 0
    epoch_gen_sync_disc_loss = 0
    epoch_gen_loss_all = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    gen.eval()
    frame_disc.eval()
    multiframe_disc.eval()
    seq_disc.eval()
    sync_disc.eval()
    r = cfg.model.reduction_factor
    lip_mean = val_dataset.lip_mean
    lip_std = val_dataset.lip_std

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        lip_len = torch.clamp(lip_len, max=lip.shape[-1])

        with torch.no_grad():
            output = gen(lip[..., 0], feature, lip_len)
            
            # frame disc
            frame_index = random.randint(0, torch.min(lip_len).item() - 1)
            fake_frame = frame_disc(output[..., frame_index].squeeze(-1), lip[..., 0])
            real_frame = frame_disc(lip[..., frame_index].squeeze(-1), lip[..., 0])
            frame_disc_loss = F.mse_loss(fake_frame, torch.zeros_like(fake_frame)) + F.mse_loss(real_frame, torch.ones_like(real_frame))

            # multiframe disc
            multiframe_start_index = random.randint(0, torch.min(lip_len).item() - int(cfg.train.multiframe_disc_input_sec * cfg.model.fps) - 1)
            fake_multiframe = multiframe_disc(output.detach()[..., multiframe_start_index:multiframe_start_index + int(cfg.train.multiframe_disc_input_sec * cfg.model.fps)])
            real_multiframe = multiframe_disc(lip[..., multiframe_start_index:multiframe_start_index + int(cfg.train.multiframe_disc_input_sec * cfg.model.fps)])
            multiframe_disc_loss = F.mse_loss(fake_multiframe, torch.zeros_like(fake_multiframe)) + F.mse_loss(real_multiframe, torch.ones_like(real_multiframe))

            # seq disc
            if torch.min(lip_len).item() - 1 - int(cfg.train.seq_disc_input_sec * cfg.model.fps) < 0:
                seq_start_index = 0
            else:
                seq_start_index = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.seq_disc_input_sec * cfg.model.fps))
            fake_seq = seq_disc(
                output[..., seq_start_index:seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)], 
                feature[..., int(seq_start_index * r):int((seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)) * r)]
            )
            real_seq = seq_disc(
                lip[..., seq_start_index:seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)], 
                feature[..., int(seq_start_index * r):int((seq_start_index + int(cfg.train.seq_disc_input_sec * cfg.model.fps)) * r)]
            )
            seq_disc_loss = F.mse_loss(fake_seq, torch.zeros_like(fake_seq)) + F.mse_loss(real_seq, torch.ones_like(real_seq))

            # sync disc
            sync_start_index = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))
            sync_start_index_fake = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))
            while sync_start_index == sync_start_index_fake:
                sync_start_index_fake = random.randint(0, torch.min(lip_len).item() - 1 - int(cfg.train.sync_disc_input_sec * cfg.model.fps))

            lip_crop = lip[..., sync_start_index:sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)]
            output_crop = output[..., sync_start_index:sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)]
            feature_crop_real = feature[..., int(sync_start_index * r):int((sync_start_index + int(cfg.train.sync_disc_input_sec * cfg.model.fps)) * r)]
            feature_crop_fake = feature[..., int(sync_start_index_fake * r):int((sync_start_index_fake + int(cfg.train.sync_disc_input_sec * cfg.model.fps)) * r)]

            fake_sync = sync_disc(output_crop, feature_crop_real)
            not_sync = sync_disc(lip_crop, feature_crop_fake)
            real_sync = sync_disc(lip_crop, feature_crop_real)
            sync_disc_loss = F.mse_loss(fake_sync, torch.zeros_like(fake_sync)) + F.mse_loss(not_sync, torch.zeros_like(not_sync))\
                + F.mse_loss(real_sync, torch.ones_like(real_sync))

            iter_cnt += 1
            if cfg.train.debug:
                if iter_cnt > cfg.train.debug_iter:
                    check_movie(lip[0], output[0], lip_mean, lip_std, cfg, "val_movie", current_time, ckpt_time)
                    break
            if iter_cnt % (all_iter - 1) == 0:
                check_movie(lip[0], output[0], lip_mean, lip_std, cfg, "val_movie", current_time, ckpt_time)

            mask = 1.0 - make_pad_mask(lip_len, lip.shape[-1]).to(torch.float32).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 1, T)
            mask = mask.to(torch.bool)
            output = torch.masked_select(output, mask)
            lip = torch.masked_select(lip, mask)
            gen_l1_loss = F.l1_loss(output, lip)

            gen_frame_disc_loss = F.mse_loss(fake_frame, torch.ones_like(fake_frame))
            gen_multiframe_disc_loss = F.mse_loss(fake_multiframe, torch.ones_like(fake_multiframe))
            gen_seq_disc_loss = F.mse_loss(fake_seq, torch.ones_like(fake_seq))
            gen_sync_disc_loss = F.mse_loss(fake_sync, torch.ones_like(fake_sync))
            gen_loss_all = cfg.train.l1_weight * gen_l1_loss + cfg.train.frame_disc_weight * gen_frame_disc_loss \
                + cfg.train.multiframe_disc_weight * gen_multiframe_disc_loss + cfg.train.seq_disc_weight * gen_seq_disc_loss \
                    + cfg.train.sync_disc_weight * gen_sync_disc_loss

        epoch_frame_disc_loss += frame_disc_loss.item()
        epoch_multiframe_disc_loss += multiframe_disc_loss.item()
        epoch_seq_disc_loss += seq_disc_loss.item()
        epoch_sync_disc_loss += sync_disc_loss.item()
        wandb.log({"val_frame_disc_loss": frame_disc_loss})
        wandb.log({"val_multiframe_disc_loss": multiframe_disc_loss})
        wandb.log({"val_seq_disc_loss": seq_disc_loss})
        wandb.log({"val_sync_disc_loss": sync_disc_loss})

        epoch_gen_l1_loss += gen_l1_loss.item()
        epoch_gen_frame_disc_loss += gen_frame_disc_loss.item()
        epoch_gen_multiframe_disc_loss += gen_multiframe_disc_loss.item()
        epoch_gen_seq_disc_loss += gen_seq_disc_loss.item()
        epoch_gen_sync_disc_loss += gen_sync_disc_loss.item()
        epoch_gen_loss_all += gen_loss_all.item()
        wandb.log({"val_gen_l1_loss": gen_l1_loss})
        wandb.log({"val_gen_frame_disc_loss": gen_frame_disc_loss})
        wandb.log({"val_gen_multiframe_disc_loss": gen_multiframe_disc_loss})
        wandb.log({"val_gen_seq_disc_loss": gen_seq_disc_loss})
        wandb.log({"val_gen_sync_disc_loss": gen_sync_disc_loss})
        wandb.log({"val_gen_loss_all": gen_loss_all})
    
    epoch_frame_disc_loss /= iter_cnt
    epoch_multiframe_disc_loss /= iter_cnt
    epoch_seq_disc_loss /= iter_cnt
    epoch_sync_disc_loss /= iter_cnt
    epoch_gen_l1_loss /= iter_cnt
    epoch_gen_frame_disc_loss /= iter_cnt
    epoch_gen_multiframe_disc_loss /= iter_cnt
    epoch_gen_seq_disc_loss /= iter_cnt
    epoch_gen_sync_disc_loss /= iter_cnt
    epoch_gen_loss_all /= iter_cnt
    return epoch_frame_disc_loss, epoch_multiframe_disc_loss,  epoch_seq_disc_loss, epoch_sync_disc_loss, \
        epoch_gen_l1_loss, epoch_gen_frame_disc_loss, epoch_gen_multiframe_disc_loss, \
            epoch_gen_seq_disc_loss, epoch_gen_sync_disc_loss, epoch_gen_loss_all


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
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader(cfg, train_data_root, val_data_root)

    train_epoch_frame_disc_loss_list = []
    train_epoch_multiframe_disc_loss_list = []
    train_epoch_seq_disc_loss_list = []
    train_epoch_sync_disc_loss_list = []
    train_epoch_gen_l1_loss_list = []
    train_epoch_gen_frame_disc_loss_list = []
    train_epoch_gen_multiframe_disc_loss_list = []
    train_epoch_gen_seq_disc_loss_list = []
    train_epoch_gen_sync_disc_loss_list = []
    train_epoch_gen_loss_all_list = []
    val_epoch_frame_disc_loss_list = []
    val_epoch_multiframe_disc_loss_list = []
    val_epoch_seq_disc_loss_list = []
    val_epoch_sync_disc_loss_list = []
    val_epoch_gen_l1_loss_list = []
    val_epoch_gen_frame_disc_loss_list = []
    val_epoch_gen_multiframe_disc_loss_list = []
    val_epoch_gen_seq_disc_loss_list = []
    val_epoch_gen_sync_disc_loss_list = []
    val_epoch_gen_loss_all_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        gen, frame_disc, multiframe_disc, seq_disc, sync_disc = make_model(cfg, device)

        optimizer_gen = torch.optim.Adam(
            params=gen.parameters(),
            lr=cfg.train.lr_gen, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )
        optimizer_frame_disc = torch.optim.Adam(
            params=frame_disc.parameters(),
            lr=cfg.train.lr_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )
        optimizer_multiframe_disc = torch.optim.Adam(
            params=multiframe_disc.parameters(),
            lr=cfg.train.lr_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )
        optimizer_seq_disc = torch.optim.Adam(
            params=seq_disc.parameters(),
            lr=cfg.train.lr_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )
        optimizer_sync_disc = torch.optim.Adam(
            params=sync_disc.parameters(),
            lr=cfg.train.lr_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )

        scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_gen, gamma=cfg.train.lr_decay_exp
        )
        scheduler_frame_disc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_frame_disc, gamma=cfg.train.lr_decay_exp
        )
        scheduler_multiframe_disc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_multiframe_disc, gamma=cfg.train.lr_decay_exp
        )
        scheduler_seq_disc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_seq_disc, gamma=cfg.train.lr_decay_exp
        )
        scheduler_sync_disc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_sync_disc, gamma=cfg.train.lr_decay_exp
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
            frame_disc.load_state_dict(checkpoint["frame_disc"])
            seq_disc.load_state_dict(checkpoint["seq_disc"])
            sync_disc.load_state_dict(checkpoint["sync_disc"])
            optimizer_gen.load_state_dict(checkpoint["optimizer_gen"])
            optimizer_frame_disc.load_state_dict(checkpoint["optimizer_frame_disc"])
            optimizer_multiframe_disc.load_state_dict(checkpoint["optimizer_multiframe_disc"])
            optimizer_seq_disc.load_state_dict(checkpoint["optimizer_seq_disc"])
            optimizer_sync_disc.load_state_dict(checkpoint["optimizer_sync_disc"])
            scheduler_gen.load_state_dict(checkpoint["scheduler_gen"])
            scheduler_frame_disc.load_state_dict(checkpoint["scheduler_frame_disc"])
            scheduler_multiframe_disc.load_state_dict(checkpoint["scheduler_multiframe_disc"])
            scheduler_seq_disc.load_state_dict(checkpoint["scheduler_seq_disc"])
            scheduler_sync_disc.load_state_dict(checkpoint["scheduler_sync_disc"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            
        if cfg.train.tts_vae_finetuning:
            print("load pretrained model for finetuning")
            checkpoint_path = Path(cfg.train.tts_vae_finetuning_pretrained_model_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            gen.load_state_dict(checkpoint["gen"])
            frame_disc.load_state_dict(checkpoint["frame_disc"])
            seq_disc.load_state_dict(checkpoint["seq_disc"])
            sync_disc.load_state_dict(checkpoint["sync_disc"])

        wandb.watch(gen, **cfg.wandb_conf.watch)
        wandb.watch(frame_disc, **cfg.wandb_conf.watch)
        wandb.watch(multiframe_disc, **cfg.wandb_conf.watch)
        wandb.watch(seq_disc, **cfg.wandb_conf.watch)
        wandb.watch(sync_disc, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            lr_gen = scheduler_gen.get_last_lr()[0]
            lr_frame_disc = scheduler_frame_disc.get_last_lr()[0]
            lr_multiframe_disc = scheduler_multiframe_disc.get_last_lr()[0]
            lr_seq_disc = scheduler_seq_disc.get_last_lr()[0]
            lr_sync_disc = scheduler_sync_disc.get_last_lr()[0]
            print(f"learning rate")
            print(f"gen = {lr_gen}, frame_disc = {lr_frame_disc}, multiframe_disc = {lr_multiframe_disc}, seq_disc = {lr_seq_disc}, sync_disc = {lr_sync_disc}")
            wandb.log({"lr_gen": lr_gen})
            wandb.log({"lr_frame_disc": lr_frame_disc})
            wandb.log({"lr_multiframe_disc": lr_multiframe_disc})
            wandb.log({"lr_seq_disc": lr_seq_disc})
            wandb.log({"lr_sync_disc": lr_sync_disc})

            epoch_frame_disc_loss, epoch_multiframe_disc_loss,  epoch_seq_disc_loss, epoch_sync_disc_loss, \
                epoch_gen_l1_loss, epoch_gen_frame_disc_loss, epoch_gen_multiframe_disc_loss, \
                    epoch_gen_seq_disc_loss, epoch_gen_sync_disc_loss, epoch_gen_loss_all = train_one_epoch(
                        gen=gen,
                        frame_disc=frame_disc,
                        multiframe_disc=multiframe_disc,
                        seq_disc=seq_disc,
                        sync_disc=sync_disc,
                        train_loader=train_loader,
                        train_dataset=train_dataset,
                        optimizer_gen=optimizer_gen,
                        optimizer_frame_disc=optimizer_frame_disc,
                        optimizer_multiframe_disc=optimizer_multiframe_disc,
                        optimizer_seq_disc=optimizer_seq_disc,
                        optimizer_sync_disc=optimizer_sync_disc,
                        device=device,
                        cfg=cfg,
                        ckpt_time=ckpt_time,
                    )

            train_epoch_frame_disc_loss_list.append(epoch_frame_disc_loss)
            train_epoch_multiframe_disc_loss_list.append(epoch_multiframe_disc_loss)
            train_epoch_seq_disc_loss_list.append(epoch_seq_disc_loss)
            train_epoch_sync_disc_loss_list.append(epoch_sync_disc_loss)
            train_epoch_gen_l1_loss_list.append(epoch_gen_l1_loss)
            train_epoch_gen_frame_disc_loss_list.append(epoch_gen_frame_disc_loss)
            train_epoch_gen_multiframe_disc_loss_list.append(epoch_gen_multiframe_disc_loss)
            train_epoch_gen_seq_disc_loss_list.append(epoch_gen_seq_disc_loss)
            train_epoch_gen_sync_disc_loss_list.append(epoch_gen_sync_disc_loss)
            train_epoch_gen_loss_all_list.append(epoch_gen_loss_all)

            epoch_frame_disc_loss, epoch_multiframe_disc_loss,  epoch_seq_disc_loss, epoch_sync_disc_loss, \
                epoch_gen_l1_loss, epoch_gen_frame_disc_loss, epoch_gen_multiframe_disc_loss, \
                    epoch_gen_seq_disc_loss, epoch_gen_sync_disc_loss, epoch_gen_loss_all = val_one_epoch(
                        gen=gen,
                        frame_disc=frame_disc,
                        multiframe_disc=multiframe_disc,
                        seq_disc=seq_disc,
                        sync_disc=sync_disc,
                        val_loader=val_loader,
                        val_dataset=val_dataset,
                        device=device,
                        cfg=cfg,
                        ckpt_time=ckpt_time,
                    )

            val_epoch_frame_disc_loss_list.append(epoch_frame_disc_loss)
            val_epoch_multiframe_disc_loss_list.append(epoch_multiframe_disc_loss)
            val_epoch_seq_disc_loss_list.append(epoch_seq_disc_loss)
            val_epoch_sync_disc_loss_list.append(epoch_sync_disc_loss)
            val_epoch_gen_l1_loss_list.append(epoch_gen_l1_loss)
            val_epoch_gen_frame_disc_loss_list.append(epoch_gen_frame_disc_loss)
            val_epoch_gen_multiframe_disc_loss_list.append(epoch_gen_multiframe_disc_loss)
            val_epoch_gen_seq_disc_loss_list.append(epoch_gen_seq_disc_loss)
            val_epoch_gen_sync_disc_loss_list.append(epoch_gen_sync_disc_loss)
            val_epoch_gen_loss_all_list.append(epoch_gen_loss_all)

            scheduler_gen.step()
            scheduler_frame_disc.step()
            scheduler_multiframe_disc.step()
            scheduler_seq_disc.step()
            scheduler_sync_disc.step()

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    gen=gen,
                    frame_disc=frame_disc,
                    seq_disc=seq_disc,
                    sync_disc=sync_disc,
                    optimizer_gen=optimizer_gen,
                    optimizer_frame_disc=optimizer_frame_disc,
                    optimizer_multiframe_disc=optimizer_multiframe_disc,
                    optimizer_seq_disc=optimizer_seq_disc,
                    optimizer_sync_disc=optimizer_sync_disc,
                    scheduler_gen=scheduler_gen,
                    scheduler_frame_disc=scheduler_frame_disc,
                    scheduler_multiframe_disc=scheduler_multiframe_disc,
                    scheduler_seq_disc=scheduler_seq_disc,
                    scheduler_sync_disc=scheduler_sync_disc,
                    train_epoch_frame_disc_loss_list=train_epoch_frame_disc_loss_list,
                    train_epoch_multiframe_disc_loss_list=train_epoch_multiframe_disc_loss_list,
                    train_epoch_seq_disc_loss_list=train_epoch_seq_disc_loss_list,
                    train_epoch_sync_disc_loss_list=train_epoch_sync_disc_loss_list,
                    train_epoch_gen_l1_loss_list=train_epoch_gen_l1_loss_list,
                    train_epoch_gen_frame_disc_loss_list=train_epoch_gen_frame_disc_loss_list,
                    train_epoch_gen_multiframe_disc_loss_list=train_epoch_gen_multiframe_disc_loss_list,
                    train_epoch_gen_seq_disc_loss_list=train_epoch_gen_seq_disc_loss_list,
                    train_epoch_gen_sync_disc_loss_list=train_epoch_gen_sync_disc_loss_list,
                    train_epoch_gen_loss_all_list=train_epoch_gen_loss_all_list,
                    val_epoch_frame_disc_loss_list=val_epoch_frame_disc_loss_list,
                    val_epoch_multiframe_disc_loss_list=val_epoch_multiframe_disc_loss_list,
                    val_epoch_seq_disc_loss_list=val_epoch_seq_disc_loss_list,
                    val_epoch_sync_disc_loss_list=val_epoch_sync_disc_loss_list,
                    val_epoch_gen_l1_loss_list=val_epoch_gen_l1_loss_list,
                    val_epoch_gen_frame_disc_loss_list=val_epoch_gen_frame_disc_loss_list,
                    val_epoch_gen_multiframe_disc_loss_list=val_epoch_gen_multiframe_disc_loss_list,
                    val_epoch_gen_seq_disc_loss_list=val_epoch_gen_seq_disc_loss_list,
                    val_epoch_gen_sync_disc_loss_list=val_epoch_gen_sync_disc_loss_list,
                    val_epoch_gen_loss_all_list=val_epoch_gen_loss_all_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt"),
                )
            save_loss(train_epoch_frame_disc_loss_list, val_epoch_frame_disc_loss_list, save_path, "frame_disc_loss")
            save_loss(train_epoch_multiframe_disc_loss_list, val_epoch_multiframe_disc_loss_list, save_path, "multiframe_disc_loss")
            save_loss(train_epoch_seq_disc_loss_list, val_epoch_seq_disc_loss_list, save_path, "seq_disc_loss")
            save_loss(train_epoch_sync_disc_loss_list, val_epoch_sync_disc_loss_list, save_path, "sync_disc_loss")
            save_loss(train_epoch_gen_l1_loss_list, val_epoch_gen_l1_loss_list, save_path, "gen_l1_loss")
            save_loss(train_epoch_gen_frame_disc_loss_list, val_epoch_gen_frame_disc_loss_list, save_path, "gen_frame_disc_loss")
            save_loss(train_epoch_gen_multiframe_disc_loss_list, val_epoch_gen_multiframe_disc_loss_list, save_path, "gen_multiframe_disc_loss")
            save_loss(train_epoch_gen_seq_disc_loss_list, val_epoch_gen_seq_disc_loss_list, save_path, "gen_seq_disc_loss")
            save_loss(train_epoch_gen_sync_disc_loss_list, val_epoch_gen_sync_disc_loss_list, save_path, "gen_sync_disc_loss")
            save_loss(train_epoch_gen_loss_all_list, val_epoch_gen_loss_all_list, save_path, "gen_loss_all")


if __name__ == "__main__":
    main()