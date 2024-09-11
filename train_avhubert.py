from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
import gc
import torch
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import (
    make_train_val_loader,
    save_loss,
    get_path_train,
    check_mel_nar,
    count_params,
    fix_random_seed,
)

from model.avhubert.avhubert import(
    Config,
    SELayer,
    DepthwiseSeparableConv,
    BasicBlock,
    ResNet,
    ResEncoder,
    FairseqDropout,
    MultiheadAttention,
    TransformerSentenceEncoderLayer,
    SamePad,
    TransformerEncoder,
    SubModel,
    AVHuBERT
)

from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="8f73812c7584e5c3a3007a1159fb3b621ba621e2")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    train_loss_list,
    train_mse_loss_list,
    train_classifier_loss_list,
    val_loss_list,
    val_mse_loss_list,
    val_classifier_loss_list,
    epoch,
    ckpt_path,
):
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'train_loss_list': train_loss_list,
        'train_mse_loss_list': train_mse_loss_list,
        'train_classifier_loss_list': train_classifier_loss_list,
        'val_loss_list': val_loss_list,
        'val_mse_loss_list': val_mse_loss_list,
        'val_classifier_loss_list': val_classifier_loss_list,
        'epoch': epoch,
    }
    if torch.cuda.is_available():
        save_dict["cuda_random"] = torch.cuda.get_rng_state()
    torch.save(save_dict, ckpt_path)

#!要編集--------------------------------------------------------------↓!!
def make_model(cfg, device):
    model = AVHuBERT(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_inner_channels=cfg.model.res_inner_channels,
        which_res=cfg.model.which_res,
        rnn_n_layers=cfg.model.rnn_n_layers,
        rnn_which_norm=cfg.model.rnn_which_norm,
        trans_n_layers=cfg.model.trans_enc_n_layers,
        trans_n_head=cfg.model.trans_enc_n_head,
        trans_pos_max_len=int(cfg.model.fps * cfg.model.input_lip_sec),
        conf_n_layers=cfg.model.conf_n_layers,
        conf_n_head=cfg.model.conf_n_head,
        conf_feedforward_expansion_factor=cfg.model.conf_feed_forward_expansion_factor,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_kernel_size=cfg.model.tc_kernel_size,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        where_spk_emb=cfg.train.where_spk_emb,
        use_spk_emb=cfg.train.use_spk_emb,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        rnn_dropout=cfg.train.rnn_dropout,
        is_large=cfg.model.is_large,
        adversarial_learning=cfg.train.adversarial_learning,
        reduction_factor=cfg.model.reduction_factor,
    )

    count_params(model, "model")
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_f,
    device,
    cfg,
    ckpt_time
):
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_classifier_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("start training") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename= batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)

        output, classifier_out, fmaps = model(lip, lip_len, spk_emb)

        mse_loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])

        if cfg.train.adversarial_learning:
            classifier_loss = loss_f.cross_entropy_loss(classifier_out, speaker_idx, ignore_index=-100)
        else:
            classifier_loss = torch.tensor(0)

        loss = cfg.train.mse_weight * mse_loss + cfg.train.classifier_weight * classifier_loss

        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        wandb.log({"train_loss": loss})
        wandb.log({"train_mse_loss": mse_loss})
        wandb.log({"train_classifier_loss": classifier_loss})

        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    return epoch_loss, epoch_mse_loss, epoch_classifier_loss


def val_one_epoch(
    model,
    val_loader,
    loss_f,
    device,
    cfg,
    ckpt_time
):
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_classifier_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("start validation")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename= batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        
        with torch.no_grad():
            output, classifier_out, fmaps = model(lip, lip_len, spk_emb)

        mse_loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])

        if cfg.train.adversarial_learning:
            classifier_loss = loss_f.cross_entropy_loss(classifier_out, speaker_idx, ignore_index=-100)
        else:
            classifier_loss = torch.tensor(0)

        loss = cfg.train.mse_weight * mse_loss + cfg.train.classifier_weight * classifier_loss

        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        wandb.log({"val_loss": loss})
        wandb.log({"val_mse_loss": mse_loss})
        wandb.log({"val_classifier_loss": classifier_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
                break

        if all_iter - 1 > 0:
            if iter_cnt % (all_iter - 1) == 0:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
        else:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    return epoch_loss, epoch_mse_loss, epoch_classifier_loss

#version_base=None, hydra.mainに入ってるやつ
@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    fix_random_seed(cfg.train.random_seed)

    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, enum_to_str=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # path
    train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"train_data_root = {train_data_root}")
    print(f"val_data_root = {val_data_root}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, train_data_root, val_data_root)

    loss_f = MaskedLoss()
    train_loss_list = []
    train_mse_loss_list = []
    train_classifier_loss_list = []
    val_loss_list = []
    val_mse_loss_list = []
    val_classifier_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.train.lr_decay_exp,
        )
        
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
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_loss_list = checkpoint["train_loss_list"]
            train_mse_loss_list = checkpoint["train_mse_loss_list"]
            train_classifier_loss_list = checkpoint["train_classifier_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]
            val_mse_loss_list = checkpoint["val_mse_loss_list"]
            val_classifier_loss_list = checkpoint["val_classifier_loss_list"]
            
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
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            # training
            epoch_loss, epoch_mse_loss, epoch_classifier_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_mse_loss_list.append(epoch_mse_loss)
            train_classifier_loss_list.append(epoch_classifier_loss)

            # validation
            epoch_loss, epoch_mse_loss, epoch_classifier_loss = val_one_epoch(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_mse_loss_list.append(epoch_mse_loss)
            val_classifier_loss_list.append(epoch_classifier_loss)

            scheduler.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_list=train_loss_list,
                    train_mse_loss_list=train_mse_loss_list,
                    train_classifier_loss_list=train_classifier_loss_list,
                    val_loss_list=val_loss_list,
                    val_mse_loss_list=val_mse_loss_list,
                    val_classifier_loss_list=val_classifier_loss_list,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{current_epoch}.ckpt"),
                )
            
            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            save_loss(train_classifier_loss_list, val_classifier_loss_list, save_path, "classifier_loss")

    wandb.finish()


if __name__=='__main__':
    main()