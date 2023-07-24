from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from timm.scheduler import CosineLRScheduler

from utils import set_config, get_path_train, make_train_val_loader_tts_face_ddp, count_params, check_mel_default, save_loss, check_movie, check_attention_weight
from model.model_tts_face_trans import TransformerFaceSpeechSynthesizer
from loss import MaskedLoss
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
    model, optimizer, scheduler, scaler,
    train_loss_list,
    train_output_loss_list,
    train_dec_output_loss_list,
    train_lip_output_loss_list,
    train_dec_lip_output_loss_list,
    train_stop_token_loss_list,
    val_loss_list,
    val_output_loss_list,
    val_dec_output_loss_list,
    val_lip_output_loss_list,
    val_dec_lip_output_loss_list,
    val_stop_token_loss_list,
    epoch, ckpt_path):
    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "scaler" : scaler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        "train_loss_list" : train_loss_list,
        "train_output_loss_list" : train_output_loss_list,
        "train_dec_output_loss_list" : train_dec_output_loss_list,
        "train_lip_output_loss_list" : train_lip_output_loss_list,
        "train_dec_lip_output_loss_list" : train_dec_lip_output_loss_list,
        "train_stop_token_loss_list" : train_stop_token_loss_list,
        "val_loss_list" : val_loss_list,
        "val_output_loss_list" : val_output_loss_list,
        "val_dec_output_loss_list" : val_dec_output_loss_list,
        "val_lip_output_loss_list" : val_lip_output_loss_list,
        "val_dec_lip_output_loss_list" : val_dec_lip_output_loss_list,
        "val_stop_token_loss_list" : val_stop_token_loss_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg):
    model = TransformerFaceSpeechSynthesizer(
        n_vocab=cfg.model.n_vocab,
        enc_n_layers=cfg.model.tft_enc_n_layers,
        enc_n_head=cfg.model.tft_enc_n_head,
        enc_d_model=cfg.model.tft_enc_d_model,
        reduction_factor=cfg.model.reduction_factor,
        enc_conv_kernel_size=cfg.model.tft_enc_conv_kernel_size,
        enc_conv_n_layers=cfg.model.tft_enc_conv_n_layers,
        enc_conv_dropout=cfg.model.tft_enc_conv_dropout,
        dec_n_layers=cfg.model.tft_dec_n_layers,
        dec_n_head=cfg.model.tft_dec_n_head,
        dec_d_model=cfg.model.tft_dec_d_model,
        feat_pre_hidden_channels=cfg.model.tft_feat_pre_hidden_channels,
        feat_pre_n_layers=cfg.model.tft_feat_pre_n_layers,
        feat_prenet_dropout=cfg.model.tft_feat_prenet_dropout,
        out_channels=cfg.model.out_channels,
        lip_channels=cfg.model.in_channels,
        lip_prenet_hidden_channels=cfg.model.tft_lip_prenet_hidden_channels,
        lip_prenet_dropout=cfg.model.tft_lip_prenet_dropout,
        lip_out_hidden_channels=cfg.model.tft_lip_out_hidden_channels,
        lip_out_dropout=cfg.model.tft_lip_out_dropout,
        feat_post_hidden_channels=cfg.model.tft_feat_post_hidden_channels,
        feat_post_n_layers=cfg.model.tft_feat_post_n_layers,
        feat_post_kernel_size=cfg.model.tft_feat_post_kernel_size,
        feat_post_dropout=cfg.model.tft_feat_post_dropout,
        lip_post_hidden_channels=cfg.model.tft_lip_post_hidden_channels,
        lip_post_n_layers=cfg.model.tft_lip_post_n_layers,
        lip_post_kernel_size=cfg.model.tft_lip_post_kernel_size,
        lip_post_dropout=cfg.model.tft_lip_post_dropout,
        which_norm=cfg.model.tft_which_norm,
    )
    count_params(model, "model")
    return model


def plot_trans_att_w(att_w_list, filename, cfg, ckpt_time):
    for i, att_w in enumerate(att_w_list):
        check_attention_weight(att_w, cfg, f"{filename}_{i}", current_time, ckpt_time)


def train_one_epoch(run, model, train_loader, train_dataset, optimizer, loss_f, scaler, rank, cfg, ckpt_time):
    epoch_loss = 0
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_lip_output_loss = 0
    epoch_dec_lip_output_loss = 0
    epoch_stop_token_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("start training")
    print(f"rank = {rank}")
    model.train()
    lip_mean = train_dataset.lip_mean
    lip_std = train_dataset.lip_std

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label = batch
        text = text.to(rank)
        feature = feature.to(rank)
        lip = lip.to(rank)
        stop_token = stop_token.to(rank)
        text_len = text_len.to(rank)
        feature_len = feature_len.to(rank)
        lip_len = lip_len.to(rank)
        spk_emb = spk_emb.to(rank)

        with autocast():
            dec_feat_output, dec_lip_output, feat_output, lip_output, logit = model(text, text_len, lip_len, feature, lip, spk_emb=spk_emb)

            dec_output_loss = loss_f.mse_loss(dec_feat_output, feature, feature_len, feature.shape[-1])
            output_loss = loss_f.mse_loss(feat_output, feature, feature_len, feature.shape[-1])

            dec_lip_output_loss = loss_f.mse_loss(dec_lip_output, lip, lip_len, lip.shape[-1])
            lip_output_loss = loss_f.mse_loss(lip_output, lip, lip_len, lip.shape[-1])

            logit_mask = 1.0 - make_pad_mask(feature_len, feature.shape[-1]).to(torch.float32).squeeze(1)
            logit_mask = logit_mask.to(torch.bool)
            logit = torch.masked_select(logit, logit_mask)
            stop_token = torch.masked_select(stop_token, logit_mask)
            stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)

            total_loss = dec_output_loss + output_loss + stop_token_loss + dec_lip_output_loss + lip_output_loss
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if rank == 0:
            epoch_loss += total_loss.item()
            epoch_output_loss += output_loss.item()
            epoch_dec_output_loss += dec_output_loss.item()
            epoch_lip_output_loss += lip_output_loss.item()
            epoch_dec_lip_output_loss += dec_lip_output_loss.item()
            epoch_stop_token_loss += stop_token_loss.item()
            run.log({"train_total_loss": total_loss})
            run.log({"train_output_loss": output_loss})
            run.log({"train_dec_output_loss": dec_output_loss})
            run.log({"train_lip_output_loss": lip_output_loss})
            run.log({"train_dec_lip_output_loss": dec_lip_output_loss})
            run.log({"train_stop_token_loss": stop_token_loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if rank == 0:
                    if cfg.model.name == "mspec80":
                        check_mel_default(feature[0], feat_output[0], dec_feat_output[0], cfg, "mel_train", current_time, ckpt_time)
                    check_movie(lip[0], lip_output[0], lip_mean, lip_std, cfg, "movie_train", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if rank == 0:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], feat_output[0], dec_feat_output[0], cfg, "mel_train", current_time, ckpt_time)
                check_movie(lip[0], lip_output[0], lip_mean, lip_std, cfg, "movie_train", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    epoch_lip_output_loss /= iter_cnt
    epoch_dec_lip_output_loss /= iter_cnt
    epoch_stop_token_loss /= iter_cnt
    return epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_lip_output_loss, epoch_dec_lip_output_loss, epoch_stop_token_loss


def val_one_epoch(run, model, val_loader, val_dataset, loss_f, scaler, rank, cfg, ckpt_time):
    epoch_loss = 0
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_lip_output_loss = 0
    epoch_dec_lip_output_loss = 0
    epoch_stop_token_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("start validation")
    print(f"rank = {rank}")
    model.eval()
    lip_mean = val_dataset.lip_mean
    lip_std = val_dataset.lip_std

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label = batch
        text = text.to(rank)
        feature = feature.to(rank)
        lip = lip.to(rank)
        stop_token = stop_token.to(rank)
        text_len = text_len.to(rank)
        feature_len = feature_len.to(rank)
        lip_len = lip_len.to(rank)
        spk_emb = spk_emb.to(rank)

        with autocast():
            with torch.no_grad():
                dec_feat_output, dec_lip_output, feat_output, lip_output, logit = model(text, text_len, lip_len, feature, lip, spk_emb=spk_emb)

            dec_output_loss = loss_f.mse_loss(dec_feat_output, feature, feature_len, feature.shape[-1])
            output_loss = loss_f.mse_loss(feat_output, feature, feature_len, feature.shape[-1])

            dec_lip_output_loss = loss_f.mse_loss(dec_lip_output, lip, lip_len, lip.shape[-1])
            lip_output_loss = loss_f.mse_loss(lip_output, lip, lip_len, lip.shape[-1])

            logit_mask = 1.0 - make_pad_mask(feature_len, feature.shape[-1]).to(torch.float32).squeeze(1)
            logit_mask = logit_mask.to(torch.bool)
            logit = torch.masked_select(logit, logit_mask)
            stop_token = torch.masked_select(stop_token, logit_mask)
            stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)

            total_loss = dec_output_loss + output_loss + stop_token_loss + dec_lip_output_loss + lip_output_loss

        if rank == 0:
            epoch_loss += total_loss.item()
            epoch_output_loss += output_loss.item()
            epoch_dec_output_loss += dec_output_loss.item()
            epoch_lip_output_loss += lip_output_loss.item()
            epoch_dec_lip_output_loss += dec_lip_output_loss.item()
            epoch_stop_token_loss += stop_token_loss.item()
            run.log({"val_total_loss": total_loss})
            run.log({"val_output_loss": output_loss})
            run.log({"val_dec_output_loss": dec_output_loss})
            run.log({"val_lip_output_loss": lip_output_loss})
            run.log({"val_dec_lip_output_loss": dec_lip_output_loss})
            run.log({"val_stop_token_loss": stop_token_loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if rank == 0:
                    if cfg.model.name == "mspec80":
                        check_mel_default(feature[0], feat_output[0], dec_feat_output[0], cfg, "mel_val", current_time, ckpt_time)
                    check_movie(lip[0], lip_output[0], lip_mean, lip_std, cfg, "movie_val", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if rank == 0:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], feat_output[0], dec_feat_output[0], cfg, "mel_val", current_time, ckpt_time)
                check_movie(lip[0], lip_output[0], lip_mean, lip_std, cfg, "movie_val", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    epoch_lip_output_loss /= iter_cnt
    epoch_dec_lip_output_loss /= iter_cnt
    epoch_stop_token_loss /= iter_cnt
    return epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_lip_output_loss, epoch_dec_lip_output_loss, epoch_stop_token_loss


def process(rank, n_gpu, cfg, run):
    dist.init_process_group("nccl", rank=rank, world_size=n_gpu)

    train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = make_train_val_loader_tts_face_ddp(rank, n_gpu, cfg, train_data_root, val_data_root)

    model = make_model(cfg).to(rank)

    if cfg.train.check_point_start:
        print("load check point")
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model"])

    if cfg.model.tft_which_norm == "bn":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.train.lr, 
        betas=(cfg.train.beta_1, cfg.train.beta_2),
        weight_decay=cfg.train.weight_decay,    
    )

    if cfg.train.use_warmup_scheduler:
        scheduler = CosineLRScheduler(
            optimizer, 
            t_initial=cfg.train.max_epoch, 
            lr_min=cfg.train.warmup_min, 
            warmup_t=cfg.train.warmup_t, 
            warmup_lr_init=cfg.train.warmup_init, 
            warmup_prefix=True,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.train.lr_decay_exp,
        )

    loss_f = MaskedLoss()
    train_loss_list = []
    train_output_loss_list = []
    train_dec_output_loss_list = []
    train_lip_output_loss_list = []
    train_dec_lip_output_loss_list = []
    train_stop_token_loss_list = []
    val_loss_list = []
    val_output_loss_list = []
    val_dec_output_loss_list = []
    val_lip_output_loss_list = []
    val_dec_lip_output_loss_list = []
    val_stop_token_loss_list = []

    scaler = GradScaler()

    last_epoch = 0
    if cfg.train.check_point_start:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        random.setstate(checkpoint["random"])
        np.random.set_state(checkpoint["np_random"])
        torch.set_rng_state(checkpoint["torch"])
        torch.random.set_rng_state(checkpoint["torch_random"])
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        train_loss_list = checkpoint["train_loss_list"]
        train_output_loss_list = checkpoint["train_output_loss_list"]
        train_dec_output_loss_list = checkpoint["train_dec_output_loss_list"]
        train_lip_output_loss_list = checkpoint["train_lip_output_loss_list"]
        train_dec_lip_output_loss_list = checkpoint["train_dec_lip_output_loss_list"]
        train_stop_token_loss_list = checkpoint["train_stop_token_loss_list"]
        val_loss_list = checkpoint["val_loss_list"]
        val_output_loss_list = checkpoint["val_output_loss_list"]
        val_dec_output_loss_list = checkpoint["val_dec_output_loss_list"]
        val_lip_output_loss_list = checkpoint["val_lip_output_loss_list"]
        val_dec_lip_output_loss_list = checkpoint["val_dec_lip_output_loss_list"]
        val_stop_token_loss_list = checkpoint["val_stop_token_loss_list"]
        last_epoch = checkpoint["epoch"]

    if rank == 0:
        run.watch(model)
    
    for epoch in range(cfg.train.max_epoch - last_epoch):
        current_epoch = 1 + epoch + last_epoch
        print(f"##### {current_epoch} #####")
        if cfg.train.use_warmup_scheduler:
            lr = scheduler.get_epoch_values(current_epoch)[0]
            print(f"learning_rate = {lr}")
        else:
            lr = scheduler.get_last_lr()[0]
            print(f"learning_rate = {lr}")
        run.log({"learning_rate": lr})

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_lip_output_loss, epoch_dec_lip_output_loss, epoch_stop_token_loss = train_one_epoch(
            run=run,
            model=model,
            train_loader=train_loader,
            train_dataset=train_dataset,
            optimizer=optimizer,
            loss_f=loss_f,
            scaler=scaler,
            rank=rank,
            cfg=cfg,
            ckpt_time=ckpt_time,
        )
        train_loss_list.append(epoch_loss)
        train_output_loss_list.append(epoch_output_loss)
        train_dec_output_loss_list.append(epoch_dec_output_loss)
        train_lip_output_loss_list.append(epoch_lip_output_loss)
        train_dec_lip_output_loss_list.append(epoch_dec_lip_output_loss)
        train_stop_token_loss_list.append(epoch_stop_token_loss)

        epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_lip_output_loss, epoch_dec_lip_output_loss, epoch_stop_token_loss = val_one_epoch(
            run=run,
            model=model,
            val_loader=val_loader,
            val_dataset=val_dataset,
            loss_f=loss_f,
            scaler=scaler,
            rank=rank,
            cfg=cfg,
            ckpt_time=ckpt_time,
        )
        val_loss_list.append(epoch_loss)
        val_output_loss_list.append(epoch_output_loss)
        val_dec_output_loss_list.append(epoch_dec_output_loss)
        val_lip_output_loss_list.append(epoch_lip_output_loss)
        val_dec_lip_output_loss_list.append(epoch_dec_lip_output_loss)
        val_stop_token_loss_list.append(epoch_stop_token_loss)

        if cfg.train.use_warmup_scheduler:
            scheduler.step(current_epoch)
        else:
            scheduler.step()

        if rank == 0:
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    train_output_loss_list=train_output_loss_list,
                    train_dec_output_loss_list=train_dec_output_loss_list,
                    train_lip_output_loss_list=train_lip_output_loss_list,
                    train_dec_lip_output_loss_list=train_dec_lip_output_loss_list,
                    train_stop_token_loss_list=train_stop_token_loss_list,
                    val_loss_list=val_loss_list,
                    val_output_loss_list=val_output_loss_list,
                    val_dec_output_loss_list=val_dec_output_loss_list,
                    val_lip_output_loss_list=val_lip_output_loss_list,
                    val_dec_lip_output_loss_list=val_dec_lip_output_loss_list,
                    val_stop_token_loss_list=val_stop_token_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            save_loss(train_loss_list, val_loss_list, save_path, "total_loss")
            save_loss(train_output_loss_list, val_output_loss_list, save_path, "output_loss")
            save_loss(train_dec_output_loss_list, val_dec_output_loss_list, save_path, "dec_output_loss")
            save_loss(train_lip_output_loss_list, val_lip_output_loss_list, save_path, "lip_output_loss")
            save_loss(train_dec_lip_output_loss_list, val_dec_lip_output_loss_list, save_path, "dec_lip_output_loss")
            save_loss(train_stop_token_loss_list, val_stop_token_loss_list, save_path, "stop_token_loss")


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)

    n_gpu = torch.cuda.device_count()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6006'

    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(
        **cfg.wandb_conf.setup, 
        config=wandb_cfg, 
        settings=wandb.Settings(start_method='fork'),
        group="DDP") as run:

        mp.spawn(
            process,
            args=(n_gpu, cfg, run),
            nprocs=n_gpu,
            join=True
        )

    


if __name__ == "__main__":
    main()