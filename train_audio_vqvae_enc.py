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
from timm.scheduler import CosineLRScheduler

from utils import count_params, set_config, get_path_train, make_train_val_loader_with_external_data, save_loss, check_mel_nar, requires_grad_change
from model.vq import VQ
from model.model_vqvae import AudioEncoder, AudioDecoder, LipEncoder
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
    audio_encoder,
    lip_encoder,
    vq,
    audio_decoder,
    optimizer,
    scheduler,
    train_loss_list,
    train_mse_loss_list,
    train_ce_loss_list,
    val_loss_list,
    val_mse_loss_list,
    val_ce_loss_list,
    epoch,
    ckpt_path,
):
    torch.save({
        'audio_encoder': audio_encoder.state_dict(),
        'lip_encoder': lip_encoder.state_dict(),
        'vq': vq.state_dict(),
        'audio_decoder': audio_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'train_loss_list': train_loss_list,
        'train_mse_loss_list': train_mse_loss_list,
        'train_ce_loss_list': train_ce_loss_list,
        'val_loss_list': val_loss_list,
        'val_mse_loss_list': val_mse_loss_list,
        'val_ce_loss_list': val_ce_loss_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    audio_encoder = AudioEncoder(
        in_channels=cfg.model.n_mel_channels,
        hidden_channels=cfg.model.audio_enc_hidden_channels,
        conv_dropout=cfg.model.audio_enc_conv_dropout,
        which_encoder=cfg.model.audio_enc_which_encoder,
        rnn_n_layers=cfg.model.audio_enc_rnn_n_layers,
        rnn_dropout=cfg.model.audio_enc_rnn_dropout,
        reduction_factor=cfg.model.reduction_factor,
        rnn_which_norm=cfg.model.rnn_which_norm,
        conf_n_layers=cfg.model.audio_enc_conf_n_Layers,
        conf_n_head=cfg.model.audio_enc_conf_n_head,
        conf_feedforward_expansion_factor=cfg.model.audio_enc_conf_feed_forward_expansion_factor,
        out_channels=cfg.model.vq_emb_dim,
    )
    lip_encoder = LipEncoder(
        which_res=cfg.model.which_res,
        in_channels=cfg.model.in_channels,
        res_inner_channels=cfg.model.res_inner_channels,
        res_dropout=cfg.train.res_dropout,
        is_large=cfg.model.is_large,
        which_encoder=cfg.model.which_encoder,
        rnn_n_layers=cfg.model.rnn_n_layers,
        rnn_dropout=cfg.train.rnn_dropout,
        reduction_factor=cfg.model.reduction_factor,
        rnn_which_norm=cfg.model.rnn_which_norm,
        conf_n_layers=cfg.model.conf_n_layers,
        conf_n_head=cfg.model.conf_n_head,
        conf_feedforward_expansion_factor=cfg.model.conf_feed_forward_expansion_factor,
        out_channels=cfg.model.vq_num_emb,
    )
    vq = VQ(
        emb_dim=cfg.model.vq_emb_dim,
        num_emb=cfg.model.vq_num_emb,
    )
    audio_decoder = AudioDecoder(
        which_decoder=cfg.model.audio_dec_which_decoder,
        hidden_channels=cfg.model.vq_emb_dim + cfg.model.spk_emb_dim + 1,
        rnn_n_layers=cfg.model.audio_dec_rnn_n_layers,
        rnn_dropout=cfg.model.audio_dec_rnn_dropout,
        reduction_factor=cfg.model.reduction_factor,
        rnn_which_norm=cfg.model.rnn_which_norm,
        conf_n_layers=cfg.model.audio_dec_conf_n_layers,
        conf_n_head=cfg.model.audio_dec_conf_n_head,
        conf_feedforward_expansion_factor=cfg.model.audio_dec_conf_feedforward_expansion_factor,
        tconv_dropout=cfg.model.audio_dec_tconv_dropout,
        out_channels=cfg.model.n_mel_channels,
    )

    count_params(audio_encoder, 'audio_encoder')
    count_params(lip_encoder, 'lip_encoder')
    count_params(vq, 'vq')
    count_params(audio_decoder, 'audio_decoder')

    audio_encoder = audio_encoder.to(device)
    lip_encoder = lip_encoder.to(device)
    vq = vq.to(device)
    audio_decoder = audio_decoder.to(device)
    return audio_encoder, lip_encoder, vq, audio_decoder


def train_one_epoch(audio_encoder, lip_encoder, vq, audio_decoder, train_loader, optimizer, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_ce_loss = 0
    epoch_mse_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print('start training')
    lip_encoder.train()
    audio_encoder.eval()
    vq.eval()
    audio_decoder.eval()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        lang_id = lang_id.to(device)

        with torch.no_grad():
            audio_enc_output = audio_encoder(feature, feature_len)
            audio_enc_output = audio_enc_output.permute(0, 2, 1)    # (B, C, T)
            quantize_audio, vq_loss, embed_idx = vq(audio_enc_output)
            quantize_audio = quantize_audio.permute(0, 2, 1)    # (B, T, C)
            feature_pred_audio = audio_decoder(quantize_audio, lip_len, spk_emb, lang_id)

        lip_enc_output = lip_encoder(lip, lip_len)
        vq_idx_pred = torch.argmax(lip_enc_output, dim=-1)
        quantize_lip = F.embedding(vq_idx_pred, vq.embed.transpose(0, 1))
        feature_pred_lip = audio_decoder(quantize_lip, lip_len, spk_emb, lang_id)

        ce_loss = F.cross_entropy(lip_enc_output.permute(0, 2, 1), embed_idx)
        mse_loss = loss_f.mse_loss(feature_pred_lip, feature_pred_audio, feature_len, feature_pred_lip.shape[-1])
        loss = ce_loss + mse_loss
        loss = loss / cfg.train.iters_to_accumulate

        loss.backward()
        
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or iter_cnt % (all_iter - 1) == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_ce_loss += ce_loss.item()
        wandb.log({'train_loss': loss})
        wandb.log({'train_mse_loss': mse_loss})
        wandb.log({'train_ce_loss': ce_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_train", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_ce_loss /= iter_cnt
    return epoch_loss, epoch_mse_loss, epoch_ce_loss


def val_one_epoch(audio_encoder, lip_encoder, vq, audio_decoder, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_ce_loss = 0
    epoch_mse_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print('start validation')
    lip_encoder.eval()
    audio_encoder.eval()
    vq.eval()
    audio_decoder.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        lang_id = lang_id.to(device)

        with torch.no_grad():
            audio_enc_output = audio_encoder(feature, feature_len)
            audio_enc_output = audio_enc_output.permute(0, 2, 1)    # (B, C, T)
            quantize_audio, vq_loss, embed_idx = vq(audio_enc_output)
            quantize_audio = quantize_audio.permute(0, 2, 1)    # (B, T, C)
            feature_pred_audio = audio_decoder(quantize_audio, lip_len, spk_emb, lang_id)

            lip_enc_output = lip_encoder(lip, lip_len)
            vq_idx_pred = torch.argmax(lip_enc_output, dim=-1)
            quantize_lip = F.embedding(vq_idx_pred, vq.embed.transpose(0, 1))
            feature_pred_lip = audio_decoder(quantize_lip, lip_len, spk_emb, lang_id)

            ce_loss = F.cross_entropy(lip_enc_output.permute(0, 2, 1), embed_idx)
            mse_loss = loss_f.mse_loss(feature_pred_lip, feature_pred_audio, feature_len, feature_pred_lip.shape[-1])
            loss = ce_loss + mse_loss

        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_ce_loss += ce_loss.item()
        wandb.log({'valid_loss': loss})
        wandb.log({'valid_mse_loss': mse_loss})
        wandb.log({'valid_ce_loss': ce_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_validation", current_time, ckpt_time)
                break

        if all_iter - 1 > 0:
            if iter_cnt % (all_iter - 1) == 0:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_validation", current_time, ckpt_time)
        else:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], feature_pred_lip[0], cfg, "mel_validation", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_ce_loss /= iter_cnt
    return epoch_loss, epoch_mse_loss, epoch_ce_loss


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
    train_mse_loss_list = []
    train_ce_loss_list = []
    val_loss_list = []
    val_mse_loss_list = []
    val_ce_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        audio_encoder, lip_encoder, vq, audio_decoder = make_model(cfg, device)

        checkpoint_path = Path(cfg.train.autoencoder_check_point_path).expanduser()
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        vq.load_state_dict(checkpoint["vq"])
        audio_decoder.load_state_dict(checkpoint["audio_decoder"])

        audio_encoder = requires_grad_change(audio_encoder, False)
        vq = requires_grad_change(vq, False)
        audio_decoder = requires_grad_change(audio_decoder, False)

        if cfg.train.which_optim == 'adam':
            optimizer = torch.optim.Adam(
                params=[
                    {'params': lip_encoder.parameters()},
                ],
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )
        elif cfg.train.which_optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params=[
                    {'params': lip_encoder.parameters()},
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

        last_epoch = 0
        if cfg.train.check_point_start:
            print("load check point")
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            audio_encoder.load_state_dict(checkpoint["audio_encoder"])
            lip_encoder.load_state_dict(checkpoint["lip_encoder"])
            vq.load_state_dict(checkpoint["vq"])
            audio_decoder.load_state_dict(checkpoint["audio_decoder"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_loss_list = checkpoint["train_loss_list"]
            train_mse_loss_list = checkpoint["train_mse_loss_list"]
            train_ce_loss_list = checkpoint["train_ce_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]
            val_mse_loss_list = checkpoint["val_mse_loss_list"]
            val_ce_loss_list = checkpoint["val_ce_loss_list"]

        if cfg.train.check_point_start_separate_save_dir:
            print("load check point (separate save dir)")
            checkpoint_path = Path(cfg.train.start_ckpt_path_separate_save_dir).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            audio_encoder.load_state_dict(checkpoint["audio_encoder"])
            lip_encoder.load_state_dict(checkpoint["lip_encoder"])
            lip_encoder.load_state_dict(checkpoint["lip_encoder"])
            vq.load_state_dict(checkpoint["vq"])
            audio_decoder.load_state_dict(checkpoint["audio_decoder"])

        wandb.watch(audio_encoder, **cfg.wandb_conf.watch)
        wandb.watch(lip_encoder, **cfg.wandb_conf.watch)
        wandb.watch(vq, **cfg.wandb_conf.watch)
        wandb.watch(audio_decoder, **cfg.wandb_conf.watch)
    
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            epoch_loss, epoch_mse_loss, epoch_ce_loss = train_one_epoch(
                audio_encoder=audio_encoder,
                lip_encoder=lip_encoder,
                vq=vq,
                audio_decoder=audio_decoder,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_mse_loss_list.append(epoch_mse_loss)
            train_ce_loss_list.append(epoch_ce_loss)

            epoch_loss, epoch_mse_loss, epoch_ce_loss = val_one_epoch(
                audio_encoder=audio_encoder,
                lip_encoder=lip_encoder,
                vq=vq,
                audio_decoder=audio_decoder,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_mse_loss_list.append(epoch_mse_loss)
            val_ce_loss_list.append(epoch_ce_loss)

            if cfg.train.which_scheduler == 'exp':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == 'warmup':
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
                scheduler.step(epoch)

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    audio_encoder=audio_encoder,
                    lip_encoder=lip_encoder,
                    vq=vq,
                    audio_decoder=audio_decoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_list=train_loss_list,
                    train_mse_loss_list=train_mse_loss_list,
                    train_ce_loss_list=train_ce_loss_list,
                    val_loss_list=val_loss_list,
                    val_mse_loss_list=val_mse_loss_list,
                    val_ce_loss_list=val_ce_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )

            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            save_loss(train_ce_loss_list, val_ce_loss_list, save_path, "ce_loss")

    wandb.finish()


if __name__ == '__main__':
    main()