from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random

import torch
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import count_params, set_config, get_path_train, make_train_val_loader_with_external_data, save_loss, check_mel_nar, requires_grad_change
from loss import MaskedLoss
from train_audio_ae import make_model
from train_audio_ae_adv import make_classifier
from model.model_ae import FeatureConverter

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def make_converter(cfg, device):
    lip2audio_converter = FeatureConverter(
        in_channels=cfg.model.ae_emb_dim,
        hidden_channels=cfg.model.converter_hidden_channels,
        n_layers=cfg.model.converter_n_layers,
        dropout=cfg.model.converter_dropout,
    )
    audio2lip_converter = FeatureConverter(
        in_channels=cfg.model.ae_emb_dim,
        hidden_channels=cfg.model.converter_hidden_channels,
        n_layers=cfg.model.converter_n_layers,
        dropout=cfg.model.converter_dropout,
    )
    count_params(lip2audio_converter, 'lip2audio_converter')
    count_params(audio2lip_converter, 'audio2lip_converter')
    lip2audio_converter = lip2audio_converter.to(device)
    audio2lip_converter = audio2lip_converter.to(device)
    return lip2audio_converter, audio2lip_converter


def save_checkpoint():
    return


def train_one_epoch(
    lip_encoder,
    audio_encoder,
    audio_decoder,
    lip2audio_converter,
    audio2lip_converter,
    audio_discriminator,
    lip_discriminator,
    train_loader,
    optimizer,
    scaler,
    loss_f,
    device,
    cfg,
    ckpt_time
):
    iter_cnt = 0
    all_iter = len(train_loader)
    print('start training')
    lip_encoder.train()
    audio_encoder.train()
    audio_decoder.train()
    lip2audio_converter.train()
    audio2lip_converter.train()
    audio_discriminator.train()
    lip_discriminator.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        lang_id = lang_id.to(device)
        video_index = torch.nonzero(is_video == 1).squeeze(-1)
        if not torch.any(is_video == 1):
            print('skip')
            iter_cnt += 1
            continue

        # update discriminator
        audio_discriminator = requires_grad_change(audio_discriminator, True)
        lip_discriminator = requires_grad_change(lip_discriminator, True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                audio_enc_output = audio_encoder(feature, feature_len)
                lip_enc_output = lip_encoder(lip[video_index], lip_len[video_index])
                audio_enc_output_from_lip = lip2audio_converter(lip_enc_output)
                lip_enc_output_from_audio = audio2lip_converter(audio_enc_output)

            disc_pred_audio_real = audio_discriminator(audio_enc_output, lip_len)
            disc_pred_audio_fake = audio_discriminator(audio_enc_output_from_lip, lip_len[video_index])
            disc_pred_lip_real = lip_discriminator(lip_enc_output, lip_len[video_index])
            disc_pred_lip_fake = lip_discriminator(lip_enc_output_from_audio, lip_len)
            ls_loss_audio_disc_real = loss_f.mse_loss(disc_pred_audio_real, torch.ones_like(disc_pred_audio_real), lip_len, disc_pred_audio_real.shape[-1])
            ls_loss_audio_disc_fake = loss_f.mse_loss(disc_pred_audio_fake, torch.zeros_like(disc_pred_audio_fake), lip_len[video_index], disc_pred_audio_fake.shape[-1])
            ls_loss_lip_disc_real = loss_f.mse_loss(disc_pred_lip_real, torch.ones_like(disc_pred_lip_real), lip_len[video_index], disc_pred_lip_real.shape[-1])
            ls_loss_lip_disc_fake = loss_f.mse_loss(disc_pred_lip_fake, torch.zeros_like(disc_pred_lip_fake), lip_len, disc_pred_lip_fake.shape[-1])
            disc_loss = ls_loss_audio_disc_real + ls_loss_audio_disc_fake + ls_loss_lip_disc_real + ls_loss_lip_disc_fake

            epoch_ls_loss_audio_disc_real += ls_loss_audio_disc_real.item()
            epoch_ls_loss_audio_disc_fake += ls_loss_audio_disc_fake.item()
            epoch_ls_loss_lip_disc_real += ls_loss_lip_disc_real.item()
            epoch_ls_loss_lip_disc_fake += ls_loss_lip_disc_fake.item()
            epoch_disc_loss += disc_loss.item()
            wandb.log({'train_ls_loss_audio_disc_real': ls_loss_audio_disc_real})
            wandb.log({'train_ls_loss_audio_disc_fake': ls_loss_audio_disc_fake})
            wandb.log({'train_ls_loss_lip_disc_real': ls_loss_lip_disc_real})
            wandb.log({'train_ls_loss_lip_disc_fake': ls_loss_lip_disc_fake})
            wandb.log({'train_disc_loss': disc_loss})

        scaler.scale(disc_loss).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # update converter
        audio_discriminator = requires_grad_change(audio_discriminator, False)
        lip_discriminator = requires_grad_change(lip_discriminator, False)
        audio_decoder = requires_grad_change(audio_decoder, False)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                audio_enc_output = audio_encoder(feature, feature_len)
                lip_enc_output = lip_encoder(lip[video_index], lip_len[video_index])

            audio_enc_output_from_lip = lip2audio_converter(lip_enc_output)
            lip_enc_output_from_audio = audio2lip_converter(audio_enc_output)
            audio_enc_output_cycle = lip2audio_converter(lip_enc_output_from_audio.detach())
            lip_enc_output_cycle = audio2lip_converter(audio_enc_output_from_lip.detach())
            feature_pred_audio_from_lip = audio_decoder(audio_enc_output, lip_len, spk_emb, lang_id)
            feature_pred_audio_from_cycle = audio_decoder(audio_enc_output_cycle, lip_len, spk_emb, lang_id)
            audio_enc_output_cycle = audio_enc_output_cycle.permute(0, 2, 1)
            lip_enc_output_cycle = lip_enc_output_cycle.permute(0, 2, 1)

            disc_pred_audio_fake = audio_discriminator(audio_enc_output_from_lip, lip_len[video_index])
            disc_pred_lip_fake = lip_discriminator(lip_enc_output_from_audio, lip_len)
            ls_loss_audio_disc_fake = loss_f.mse_loss(disc_pred_audio_fake, torch.ones_like(disc_pred_audio_fake), lip_len[video_index], disc_pred_audio_fake.shape[-1])
            ls_loss_lip_disc_fake = loss_f.mse_loss(disc_pred_lip_fake, torch.ones_like(disc_pred_lip_fake), lip_len, disc_pred_lip_fake.shape[-1])
            mse_loss_audio_cycle = loss_f.mse_loss(audio_enc_output_cycle, audio_enc_output, lip_len, audio_enc_output.shape[-1])
            mse_loss_lip_cycle = loss_f.mse_loss(lip_enc_output_cycle, lip_enc_output, lip_len, lip_enc_output.shape[-1])
            mse_loss_audio_from_lip_mel = loss_f.mse_loss(feature_pred_audio_from_lip, feature, feature_len, feature_pred_audio_from_lip.shape[-1])
            mse_loss_audio_from_cycle_mel = loss_f.mse_loss(feature_pred_audio_from_cycle, feature, feature_len, feature_pred_audio_from_cycle.shape[-1])
            converter_loss = ls_loss_audio_disc_fake + ls_loss_lip_disc_fake + mse_loss_audio_cycle + mse_loss_lip_cycle \
                + mse_loss_audio_from_lip_mel + mse_loss_audio_from_cycle_mel

            epoch_disc_pred_audio_fake += disc_pred_audio_fake.item()
            epoch_disc_pred_lip_fake += disc_pred_lip_fake.item()
            epoch_ls_loss_audio_disc_fake += ls_loss_audio_disc_fake.item()
            epoch_ls_loss_lip_disc_fake += ls_loss_lip_disc_fake.item()
            epoch_mse_loss_audio_cycle += mse_loss_audio_cycle.item()
            epoch_mse_loss_lip_cycle += mse_loss_lip_cycle.item()
            epoch_mse_loss_audio_from_lip_mel += mse_loss_audio_from_lip_mel.item()
            epoch_mse_loss_audio_from_cycle_mel += mse_loss_audio_from_cycle_mel.item()
            epoch_converter_loss += converter_loss.item()
            wandb.log({'train_disc_pred_audio_fake': disc_pred_audio_fake})
            wandb.log({'train_disc_pred_lip_fake': disc_pred_lip_fake})
            wandb.log({'train_ls_loss_audio_disc_fake': ls_loss_audio_disc_fake})
            wandb.log({'train_ls_loss_lip_disc_fake': ls_loss_lip_disc_fake})
            wandb.log({'train_mse_loss_audio_cycle': mse_loss_audio_cycle})
            wandb.log({'train_mse_loss_lip_cycle': mse_loss_lip_cycle})
            wandb.log({'train_mse_loss_audio_from_lip_mel': mse_loss_audio_from_lip_mel})
            wandb.log({'train_mse_loss_audio_from_cycle_mel': mse_loss_audio_from_cycle_mel})
            wandb.log({'train_converter_loss': converter_loss})

        scaler.scale(converter_loss).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # update audio autoencoder
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            audio_enc_output = audio_encoder(feature, feature_len)
            feature_pred_audio = audio_decoder(audio_enc_output, lip_len, spk_emb, lang_id)

            mse_loss_mel_audio_ae = loss_f.mse_loss(feature_pred_audio, feature, feature_len, feature_pred_audio.shape[-1])
            bce_loss_audio = loss_f.bce_with_logits_loss(clf_pred_audio, torch.zeros_like(clf_pred_audio), lip_len, clf_pred_audio.shape[-1])
            loss_audio_ae = mse_loss_mel_audio_ae * cfg.train.mse_loss_mel_weight +\
                bce_loss_audio * cfg.train.bce_loss_weight

            epoch_loss_audio_ae += loss_audio_ae.item()
            epoch_mse_loss_mel_audio_ae += mse_loss_mel_audio_ae.item()
            epoch_bce_loss_audio_ae += bce_loss_audio.item()
            wandb.log({'train_loss_audio_ae': loss_audio_ae})
            wandb.log({'train_mse_loss_mel_audio_ae': mse_loss_mel_audio_ae})
            wandb.log({'train_bce_loss_audio_ae' : bce_loss_audio})

            loss_audio_ae = loss_audio_ae / cfg.train.iters_to_accumulate

        scaler.scale(loss_audio_ae).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # update lip encoder
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            lip_enc_output = lip_encoder(lip[video_index], lip_len[video_index])
            feature_pred_lip = audio_decoder(lip_enc_output, lip_len[video_index], spk_emb[video_index], lang_id[video_index])
            clf_pred_lip = domain_classifier(lip_enc_output, lip_len[video_index])

            mse_loss_mel_lip = loss_f.mse_loss(feature_pred_lip, feature[video_index], feature_len[video_index], feature_pred_lip.shape[-1])
            bce_loss_lip = loss_f.bce_with_logits_loss(clf_pred_lip, torch.ones_like(clf_pred_lip), lip_len[video_index], clf_pred_lip.shape[-1])
            loss_lip = mse_loss_mel_lip * cfg.train.mse_loss_mel_weight +\
                bce_loss_lip * cfg.train.bce_loss_weight
            
            epoch_loss_lip += loss_lip.item()
            epoch_mse_loss_mel_lip += mse_loss_mel_lip.item()
            epoch_bce_loss_lip += bce_loss_lip.item()
            wandb.log({'train_loss_lip': loss_lip})
            wandb.log({'train_mse_loss_mel_lip': mse_loss_mel_lip})
            wandb.log({'train_bce_loss_lip' : bce_loss_lip})

            loss_lip = loss_lip / cfg.train.iters_to_accumulate

        scaler.scale(loss_lip).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], feature_pred_audio[0], cfg, "mel_audio_train", current_time, ckpt_time)
                    check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_lip_train", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], feature_pred_audio[0], cfg, "mel_audio_train", current_time, ckpt_time)
                check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_lip_train", current_time, ckpt_time)
    return


def val_one_epoch():
    return


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
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_with_external_data(cfg, train_data_root, val_data_root)

    loss_f = MaskedLoss()

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        lip_encoder, audio_encoder, audio_decoder = make_model(cfg, device)
        audio_discriminator = make_classifier(cfg, device)
        lip_discriminator = make_classifier(cfg, device)
        lip2audio_converter, audio2lip_converter = make_converter(cfg, device)

        if cfg.train.load_pretrained_model:
            pretrained_model_path = Path(cfg.train.pretrained_model_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(pretrained_model_path)
            else:
                checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
            lip_encoder.load_state_dict(checkpoint["lip_encoder"])
            audio_encoder.load_state_dict(checkpoint["audio_encoder"])
            audio_decoder.load_state_dict(checkpoint["audio_decoder"])

        if cfg.train.which_optim == 'adam':
            optimizer = torch.optim.Adam(
                params=[
                    {'params': lip_encoder.parameters()},
                    {'params': audio_encoder.parameters()},
                    {'params': audio_decoder.parameters()},
                    {'params': audio_discriminator.parameters()},
                    {'params': lip_discriminator.parameters()},
                    {'params': lip2audio_converter.parameters()},
                    {'params': audio2lip_converter.parameters()},
                ],
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )
        elif cfg.train.which_optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params=[
                    {'params': lip_encoder.parameters()},
                    {'params': audio_encoder.parameters()},
                    {'params': audio_decoder.parameters()},
                    {'params': audio_discriminator.parameters()},
                    {'params': lip_discriminator.parameters()},
                    {'params': lip2audio_converter.parameters()},
                    {'params': audio2lip_converter.parameters()},
                ],
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
            lip_encoder.load_state_dict(checkpoint["lip_encoder"])
            audio_encoder.load_state_dict(checkpoint["audio_encoder"])
            audio_decoder.load_state_dict(checkpoint["audio_decoder"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_bce_loss_audio_clf_list = checkpoint['train_bce_loss_audio_clf_list']
            train_bce_loss_lip_clf_list = checkpoint['train_bce_loss_lip_clf_list']
            train_bce_loss_clf_list = checkpoint['train_bce_loss_clf_list']
            train_loss_audio_ae_list = checkpoint['train_loss_audio_ae_list']
            train_mse_loss_mel_audio_ae_list = checkpoint['train_mse_loss_mel_audio_ae_list']
            train_bce_loss_audio_ae_list = checkpoint['train_bce_loss_audio_ae_list']
            train_loss_lip_list = checkpoint['train_loss_lip_list']
            train_mse_loss_mel_lip_list = checkpoint['train_mse_loss_mel_lip_list']
            train_bce_loss_lip_list = checkpoint['train_bce_loss_lip_list']
            val_bce_loss_audio_clf_list = checkpoint['val_bce_loss_audio_clf_list']
            val_bce_loss_lip_clf_list = checkpoint['val_bce_loss_lip_clf_list']
            val_bce_loss_clf_list = checkpoint['val_bce_loss_clf_list']
            val_loss_audio_ae_list = checkpoint['val_loss_audio_ae_list']
            val_mse_loss_mel_audio_ae_list = checkpoint['val_mse_loss_mel_audio_ae_list']
            val_bce_loss_audio_ae_list = checkpoint['val_bce_loss_audio_ae_list']
            val_loss_lip_list = checkpoint['val_loss_lip_list']
            val_mse_loss_mel_lip_list = checkpoint['val_mse_loss_mel_lip_list']
            val_bce_loss_lip_list = checkpoint['val_bce_loss_lip_list']

        if len(cfg.train.module_is_fixed) != 0:
            print(f'\n--- Fix model parameters ---')
            if 'lip_encoder' in cfg.train.module_is_fixed:
                lip_encoder = requires_grad_change(lip_encoder, False)
            if 'audio_encoder' in cfg.train.module_is_fixed:
                audio_encoder = requires_grad_change(audio_encoder, False)
            if 'audio_decoder' in cfg.train.module_is_fixed:
                audio_decoder = requires_grad_change(audio_decoder, False)
            
            print(cfg.train.module_is_fixed)
            count_params(lip_encoder, 'lip_encoder')
            count_params(audio_encoder, 'audio_encoder')
            count_params(audio_decoder, 'audio_decoder')
            print()

        wandb.watch(lip_encoder, **cfg.wandb_conf.watch)
        wandb.watch(audio_encoder, **cfg.wandb_conf.watch)
        wandb.watch(audio_decoder, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            
            train_one_epoch(
                lip_encoder=lip_encoder,
                audio_encoder=audio_encoder,
                audio_decoder=audio_decoder,
                lip2audio_converter=lip2audio_converter,
                audio2lip_converter=audio2lip_converter,
                audio_discriminator=audio_discriminator,
                lip_discriminator=lip_discriminator,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )


if __name__ == '__main__':
    main()