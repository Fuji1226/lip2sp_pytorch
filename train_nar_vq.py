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

import torch
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import make_train_val_loader, save_loss, get_path_train, check_mel_nar, count_params, set_config, calc_class_balance
from model.model_nar_vq import Lip2SP_NARVQ
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    model = Lip2SP_NARVQ(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        norm_type=cfg.model.norm_type_lip,
        separate_frontend=cfg.train.separate_frontend,
        which_res=cfg.model.which_res,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        conformer_conv_kernel_size=cfg.model.conformer_conv_kernel_size,
        rnn_hidden_channels=cfg.model.rnn_hidden_channels,
        rnn_n_layers=cfg.model.rnn_n_layers,
        vq_emb_dim=cfg.model.vq_emb_dim,
        vq_num_emb=cfg.model.vq_num_emb,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_inner_channels=cfg.model.tc_inner_channels,
        dec_kernel_size=cfg.model.tc_kernel_size,
        tc_n_attn_layer=cfg.model.tc_n_attn_layer,
        tc_n_head=cfg.model.tc_n_head,
        tc_d_model=cfg.model.tc_d_model,
        feat_add_channels=cfg.model.tc_feat_add_channels,
        feat_add_layers=cfg.model.tc_feat_add_layers,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        apply_first_bn=cfg.train.apply_first_bn,
        use_feat_add=cfg.train.use_feat_add,
        phoneme_classes=cfg.model.n_classes,
        use_phoneme=cfg.train.use_phoneme,
        use_dec_attention=cfg.train.use_dec_attention,
        upsample_method=cfg.train.upsample_method,
        compress_rate=cfg.train.compress_rate,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        reduction_factor=cfg.model.reduction_factor,
        use_gc=cfg.train.use_gc,
    )

    count_params(model, "model")
    count_params(model.ResNet_GAP, "ResNet")
    count_params(model.encoder, "encoder")
    count_params(model.decoder, "decoder")
    count_params(model.decoder.interp_layer, "dec_interp_layer")
    count_params(model.decoder.conv_layers, "dec_conv_layers")
    count_params(model.decoder.out_layer, "dec_out_layer")
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_vq_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        if cfg.train.use_gc:
            output, feat_add_out, phoneme, vq_loss = model(lip=lip, data_len=data_len, gc=speaker)
        else:
            output, feat_add_out, phoneme, vq_loss = model(lip=lip, data_len=data_len)
        B, C, T = output.shape

        mse_loss = loss_f.mse_loss(output, feature, data_len, max_len=T, speaker=speaker)
        loss = mse_loss + vq_loss
        epoch_mse_loss += mse_loss.item()
        epoch_vq_loss += vq_loss.item()
        epoch_loss += loss.item()
        wandb.log({"train_mse_loss": mse_loss})
        wandb.log({"train_vq_loss": vq_loss})
        wandb.log({"train_loss": loss})
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
    epoch_vq_loss /= iter_cnt
    return epoch_loss, epoch_mse_loss, epoch_vq_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_vq_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("\ncalc val loss")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)
        
        with torch.no_grad():
            if cfg.train.use_gc:
                output, feat_add_out, phoneme, vq_loss = model(lip=lip, data_len=data_len, gc=speaker)
            else:
                output, feat_add_out, phoneme, vq_loss = model(lip=lip, data_len=data_len)

        B, C, T = output.shape

        mse_loss = loss_f.mse_loss(output, feature, data_len, max_len=T, speaker=speaker)
        loss = mse_loss + vq_loss
        epoch_mse_loss += mse_loss.item()
        epoch_vq_loss += vq_loss.item()
        epoch_loss += loss.item()
        wandb.log({"val_mse_loss": mse_loss})
        wandb.log({"val_vq_loss": vq_loss})
        wandb.log({"val_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_vq_loss /= iter_cnt
    return epoch_loss, epoch_mse_loss, epoch_vq_loss


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
    train_data_root, val_data_root, stat_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"train_data_root = {train_data_root}")
    print(f"val_data_root = {val_data_root}")
    print(f"stat_path = {stat_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, train_data_root, val_data_root, stat_path)

    # 損失関数
    if len(cfg.train.speaker) > 1:
        class_weight = calc_class_balance(cfg, train_data_root, device)
    else:
        class_weight = None
    loss_f = MaskedLoss(weight=class_weight, use_weighted_mse=cfg.train.use_weighted_mse)

    train_loss_list = []
    train_mse_loss_list = []
    train_vq_loss_list = []
    val_loss_list = []
    val_mse_loss_list = []
    val_vq_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.train.multi_lr_decay_step,
            gamma=cfg.train.lr_decay_rate,
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            checkpoint_path = cfg.train.start_ckpt_path
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)
    
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")
            # print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")

            # training
            epoch_loss, epoch_mse_loss, epoch_vq_loss = train_one_epoch(
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
            train_vq_loss_list.append(epoch_vq_loss)

            # validation
            epoch_loss, epoch_mse_loss, epoch_vq_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_mse_loss_list.append(epoch_mse_loss)
            val_vq_loss_list.append(epoch_vq_loss)
        
            # 学習率の更新
            scheduler.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            save_loss(train_vq_loss_list, val_vq_loss_list, save_path, "vq_loss")
                
        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()