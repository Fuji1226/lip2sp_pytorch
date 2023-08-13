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

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    lip_encoder,
    audio_encoder,
    audio_decoder,
    optimizer,
    scheduler,
    scaler,
    train_loss_list,
    train_mse_loss_mel_list,
    train_mse_loss_enc_feature_list,
    train_mse_loss_mel_lip_list,
    val_loss_list,
    val_mse_loss_mel_list,
    val_mse_loss_enc_feature_list,
    val_mse_loss_mel_lip_list,
    epoch,
    ckpt_path,
):
    torch.save({
        'lip_encoder': lip_encoder.state_dict(),
        'audio_encoder': audio_encoder.state_dict(),
        'audio_decoder': audio_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'train_loss_list': train_loss_list,
        'train_mse_loss_mel_list': train_mse_loss_mel_list,
        'train_mse_loss_enc_feature_list': train_mse_loss_enc_feature_list,
        'train_mse_loss_mel_lip_list': train_mse_loss_mel_lip_list,
        'val_loss_list': val_loss_list,
        'val_mse_loss_mel_list': val_mse_loss_mel_list,
        'val_mse_loss_enc_feature_list': val_mse_loss_enc_feature_list,
        'val_mse_loss_mel_lip_list': val_mse_loss_mel_lip_list,
        'epoch': epoch
    }, ckpt_path)


def train_one_epoch(lip_encoder, audio_encoder, audio_decoder, train_loader, optimizer, scaler, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_mse_loss_mel = 0
    epoch_mse_loss_enc_feature = 0
    epoch_mse_loss_mel_lip = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print('start training')
    lip_encoder.train()
    audio_encoder.train()
    audio_decoder.train()

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

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            lip_enc_output = lip_encoder(lip[video_index], lip_len[video_index])
            audio_enc_output = audio_encoder(feature, feature_len)
            feature_pred_audio = audio_decoder(audio_enc_output, lip_len, spk_emb, lang_id)
            with torch.no_grad():
                feature_pred_lip = audio_decoder(lip_enc_output, lip_len[video_index], spk_emb[video_index], lang_id[video_index])

            mse_loss_mel = loss_f.mse_loss(feature_pred_audio, feature, feature_len, feature_pred_audio.shape[-1])
            mse_loss_enc_feature = loss_f.mse_loss(lip_enc_output.permute(0, 2, 1), audio_enc_output[video_index].permute(0, 2, 1), lip_len[video_index], lip_enc_output.shape[1])
            mse_loss_mel_lip = loss_f.mse_loss(feature_pred_lip, feature[video_index], feature_len[video_index], feature_pred_lip.shape[-1])

            loss = mse_loss_mel * cfg.train.mse_loss_mel_weight +\
                mse_loss_enc_feature * cfg.train.mse_loss_enc_feature

            epoch_loss += loss.item()
            epoch_mse_loss_mel += mse_loss_mel.item()
            epoch_mse_loss_enc_feature += mse_loss_enc_feature.item()
            epoch_mse_loss_mel_lip += mse_loss_mel_lip.item()
            wandb.log({'train_loss': loss})
            wandb.log({'train_mse_loss_mel': mse_loss_mel})
            wandb.log({'train_mse_loss_enc_feature': mse_loss_enc_feature})
            wandb.log({'train_mse_loss_mel_lip': mse_loss_enc_feature})

            loss = loss / cfg.train.iters_to_accumulate

        scaler.scale(loss).backward()

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

    epoch_loss /= iter_cnt
    epoch_mse_loss_mel /= iter_cnt
    epoch_mse_loss_enc_feature /= iter_cnt
    epoch_mse_loss_mel_lip /= iter_cnt
    return epoch_loss, epoch_mse_loss_mel, epoch_mse_loss_enc_feature, epoch_mse_loss_mel_lip


def val_one_epoch(lip_encoder, audio_encoder, audio_decoder, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_mse_loss_mel = 0
    epoch_mse_loss_enc_feature = 0
    epoch_mse_loss_mel_lip = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print('start validation')
    lip_encoder.eval()
    audio_encoder.eval()
    audio_decoder.eval()

    for batch in val_loader:
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

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                lip_enc_output = lip_encoder(lip[video_index], lip_len[video_index])
                audio_enc_output = audio_encoder(feature, feature_len)
                feature_pred_audio = audio_decoder(audio_enc_output, lip_len, spk_emb, lang_id)
                feature_pred_lip = audio_decoder(lip_enc_output, lip_len[video_index], spk_emb[video_index], lang_id[video_index])

                mse_loss_mel = loss_f.mse_loss(feature_pred_audio, feature, feature_len, feature_pred_audio.shape[-1])
                mse_loss_enc_feature = loss_f.mse_loss(lip_enc_output.permute(0, 2, 1), audio_enc_output[video_index].permute(0, 2, 1), lip_len[video_index], lip_enc_output.shape[1])
                mse_loss_mel_lip = loss_f.mse_loss(feature_pred_lip, feature, feature_len, feature_pred_lip.shape[-1])

                loss = mse_loss_mel * cfg.train.mse_loss_mel_weight +\
                    mse_loss_enc_feature * cfg.train.mse_loss_enc_feature

                epoch_loss += loss.item()
                epoch_mse_loss_mel += mse_loss_mel.item()
                epoch_mse_loss_enc_feature += mse_loss_enc_feature.item()
                epoch_mse_loss_mel_lip += mse_loss_mel_lip.item()
                wandb.log({'val_loss': loss})
                wandb.log({'val_mse_loss_mel': mse_loss_mel})
                wandb.log({'val_mse_loss_enc_feature': mse_loss_enc_feature})
                wandb.log({'val_mse_loss_mel_lip': mse_loss_mel_lip})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], feature_pred_audio[0], cfg, "mel_audio_validation", current_time, ckpt_time)
                    check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_lip_validation", current_time, ckpt_time)
                break

        if all_iter - 1 > 0:
            if iter_cnt % (all_iter - 1) == 0:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], feature_pred_audio[0], cfg, "mel_audio_validation", current_time, ckpt_time)
                    check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_lip_validation", current_time, ckpt_time)
        else:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], feature_pred_audio[0], cfg, "mel_audio_validation", current_time, ckpt_time)
                check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_lip_validation", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    epoch_mse_loss_mel /= iter_cnt
    epoch_mse_loss_enc_feature /= iter_cnt
    epoch_mse_loss_mel_lip /= iter_cnt
    return epoch_loss, epoch_mse_loss_mel, epoch_mse_loss_enc_feature, epoch_mse_loss_mel_lip


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
    train_loss_list = []
    train_mse_loss_mel_list = []
    train_mse_loss_enc_feature_list = []
    train_mse_loss_mel_lip_list = []
    val_loss_list = []
    val_mse_loss_mel_list = []
    val_mse_loss_enc_feature_list = []
    val_mse_loss_mel_lip_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        lip_encoder, audio_encoder, audio_decoder = make_model(cfg, device)

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
                    {'params': audio_decoder.parameters()}
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
                    {'params': audio_decoder.parameters()}
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
            train_loss_list = checkpoint["train_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]

        if len(cfg.train.module_is_fixed) != 0:
            print(f'\n--- Fix model parameters ---')
            if 'lip_encoder' in cfg.train.module_is_fixed:
                requires_grad_change(lip_encoder, False)
            if 'audio_encoder' in cfg.train.module_is_fixed:
                requires_grad_change(audio_encoder, False)
            if 'audio_decoder' in cfg.train.module_is_fixed:
                requires_grad_change(audio_decoder, False)
            
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

            epoch_loss, epoch_mse_loss_mel, epoch_mse_loss_enc_feature, epoch_mse_loss_mel_lip = train_one_epoch(
                lip_encoder=lip_encoder,
                audio_encoder=audio_encoder,
                audio_decoder=audio_decoder,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_mse_loss_mel_list.append(epoch_mse_loss_mel)
            train_mse_loss_enc_feature_list.append(epoch_mse_loss_enc_feature)
            train_mse_loss_mel_lip_list.append(epoch_mse_loss_mel_lip)

            epoch_loss, epoch_mse_loss_mel, epoch_mse_loss_enc_feature, epoch_mse_loss_mel_lip = val_one_epoch(
                lip_encoder=lip_encoder,
                audio_encoder=audio_encoder,
                audio_decoder=audio_decoder,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_mse_loss_mel_list.append(epoch_mse_loss_mel)
            val_mse_loss_enc_feature_list.append(epoch_mse_loss_enc_feature)
            val_mse_loss_mel_lip_list.append(epoch_mse_loss_mel_lip)

            if cfg.train.which_scheduler == 'exp':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == 'warmup':
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
                scheduler.step(epoch)

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    lip_encoder=lip_encoder,
                    audio_encoder=audio_encoder,
                    audio_decoder=audio_decoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    train_mse_loss_mel_list=train_mse_loss_mel_list,
                    train_mse_loss_enc_feature_list=train_mse_loss_enc_feature_list,
                    train_mse_loss_mel_lip_list=train_mse_loss_mel_lip_list,
                    val_loss_list=val_loss_list,
                    val_mse_loss_mel_list=val_mse_loss_mel_list,
                    val_mse_loss_enc_feature_list=val_mse_loss_enc_feature_list,
                    val_mse_loss_mel_lip_list=val_mse_loss_mel_lip_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )

            save_loss(train_loss_list, val_loss_list, save_path, 'loss')
            save_loss(train_mse_loss_mel_list, val_mse_loss_mel_list, save_path, 'mel_loss')
            save_loss(train_mse_loss_enc_feature_list, val_mse_loss_enc_feature_list, save_path, 'enc_feature_loss')
            save_loss(train_mse_loss_mel_lip_list, val_mse_loss_mel_lip_list, save_path, 'mel_lip_loss')

    wandb.finish()


if __name__ == '__main__':
    main()