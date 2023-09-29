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

from model.model_vae import Lip2SP_VAE
from loss import MaskedLoss
from train_default import make_train_val_loader, check_feat_add, save_loss
from train_nar import check_mel
from model.mi_estimater import MyCLUBSample

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


# def save_checkpoint(model, mi_estimater, optimizer, optimizer_mi, scheduler, epoch, ckpt_path):
def save_checkpoint(model, mi_estimater, optimizer, optimizer_mi, epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'mi_estimater': mi_estimater.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_mi': optimizer_mi.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    model = Lip2SP_VAE(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        conformer_conv_kernel_size=cfg.model.conformer_conv_kernel_size,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_inner_channels=cfg.model.tc_inner_channels,
        dec_kernel_size=cfg.model.tc_kernel_size,
        feat_add_channels=cfg.model.tc_feat_add_channels,
        feat_add_layers=cfg.model.tc_feat_add_layers,
        vae_emb_dim=cfg.model.vae_emb_dim,    
        spk_emb_dim=cfg.model.spk_emb_dim,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        apply_first_bn=cfg.train.apply_first_bn,
        use_feat_add=cfg.train.use_feat_add,
        phoneme_classes=cfg.model.n_classes,
        use_phoneme=cfg.train.use_phoneme,
        upsample_method=cfg.train.upsample_method,
        compress_rate=cfg.train.compress_rate,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        reduction_factor=cfg.model.reduction_factor,
        use_gc=cfg.train.use_gc,
    )
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, mi_estimater, train_loader, optimizer, optimizer_mi, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    epoch_mse_loss = 0
    epoch_kl_loss = 0
    epoch_mi_loss = 0
    epoch_estimater_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()
    mi_estimater.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        ### update mi estimater ###
        with torch.no_grad():
            output_feat, feat_add_out_feat, phoneme_feat, output_lip, feat_add_out_lip, phoneme_lip, mu_lip, logvar_lip, z_lip, mu_feat, logvar_feat, z_feat, spk_emb = model(lip=lip, feature=feature, feat_add=feat_add, data_len=data_len)
        
        estimater_loss = - mi_estimater.loglikeli(spk_emb, z_feat)
        estimater_loss.backward()
        clip_grad_norm_(mi_estimater.parameters(), cfg.train.max_norm)
        optimizer_mi.step()
        optimizer_mi.zero_grad()

        epoch_estimater_loss += estimater_loss.item()
        wandb.log({"train_estimater_loss": estimater_loss})

        ### update model ###
        output_feat, feat_add_out_feat, phoneme_feat, output_lip, feat_add_out_lip, phoneme_lip, mu_lip, logvar_lip, z_lip, mu_feat, logvar_feat, z_feat, spk_emb = model(lip=lip, feature=feature, feat_add=feat_add, data_len=data_len)
        B, C, T = output_feat.shape

        loss = 0
        if cfg.train.use_feat_add:
            loss_feat_add = loss_f.mse_loss(feat_add_out_feat, feat_add, data_len, max_len=T)
            loss += loss_feat_add
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"train_loss_feat_add": loss_feat_add})

        mse_loss = loss_f.mse_loss(output_feat, feature, data_len, max_len=T)

        kl_loss = - 0.5 * torch.sum(1 + logvar_lip - logvar_feat - ((logvar_lip.exp() + (mu_lip - mu_feat)**2) / logvar_feat.exp()), dim=-1)
        kl_loss = torch.mean(kl_loss)

        with torch.no_grad():
            mi_loss = mi_estimater(spk_emb, z_feat)

        loss = loss + mse_loss + cfg.train.kl_weight * kl_loss + cfg.train.mi_weight * mi_loss 
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_mse_loss += mse_loss.item()
        epoch_mi_loss += mi_loss.item()
        epoch_loss += loss.item()
        epoch_kl_loss += kl_loss.item()
        wandb.log({"train_mse_loss": mse_loss})
        wandb.log({"train_mi_loss": mi_loss})
        wandb.log({"train_total_loss": loss})
        wandb.log({"train_kl_loss": kl_loss})

        # ### update lip encoder ###
        # with torch.no_grad():
        #     _, _, _, _, mu_target, logvar_target, _, _ = model(feature=feature, data_len=data_len)

        # output_lip, feat_add_out_lip, phoneme, spk_emb, mu, logvar, z, enc_output = model(lip=lip, feature=feature, data_len=data_len)

        # kl_loss = - 0.5 * torch.sum(1 + logvar - logvar_target - ((logvar.exp() + (mu - mu_target)**2) / logvar_target.exp()), dim=-1)
        # kl_loss = torch.mean(kl_loss)
        # kl_loss.backward()
        # clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        # optimizer.step()
        # optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output_feat[0], cfg, "mel_train_feat", ckpt_time)
                    check_mel(feature[0], output_lip[0], cfg, "mel_train_lip", ckpt_time)
                    if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out_feat[0], cfg, "feat_add_train_feat", ckpt_time)
                        check_feat_add(feature[0], feat_add_out_lip[0], cfg, "feat_add_train_lip", ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output_feat[0], cfg, "mel_train_feat", ckpt_time)
                check_mel(feature[0], output_lip[0], cfg, "mel_train_lip", ckpt_time)
                if cfg.train.use_feat_add:
                    check_feat_add(feature[0], feat_add_out_feat[0], cfg, "feat_add_train_feat", ckpt_time)
                    check_feat_add(feature[0], feat_add_out_lip[0], cfg, "feat_add_train_lip", ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_kl_loss /= iter_cnt
    epoch_mi_loss /= iter_cnt
    epoch_estimater_loss /= iter_cnt
    return epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_kl_loss, epoch_mi_loss, epoch_estimater_loss


def val_one_epoch(model, mi_estimater, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    epoch_mse_loss = 0
    epoch_kl_loss = 0
    epoch_mi_loss = 0
    epoch_estimater_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start") 
    model.eval()
    mi_estimater.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        with torch.no_grad():
            output_feat, feat_add_out_feat, phoneme_feat, output_lip, feat_add_out_lip, phoneme_lip, mu_lip, logvar_lip, z_lip, mu_feat, logvar_feat, z_feat, spk_emb = model(lip=lip, feature=feature, feat_add=feat_add, data_len=data_len)

            estimater_loss = - mi_estimater.loglikeli(spk_emb, z_feat)
            mi_loss = mi_estimater(spk_emb, z_feat)
        B, C, T = output_feat.shape

        loss = 0
        if cfg.train.use_feat_add:
            loss_feat_add = loss_f.mse_loss(feat_add_out_feat, feat_add, data_len, max_len=T)
            loss += loss_feat_add
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"val_loss_feat_add": loss_feat_add})

        mse_loss = loss_f.mse_loss(output_feat, feature, data_len, max_len=T)

        loss = loss + mse_loss + cfg.train.mi_weight * mi_loss

        kl_loss = - 0.5 * torch.sum(1 + logvar_lip - logvar_feat - ((logvar_lip.exp() + (mu_lip - mu_feat)**2) / logvar_feat.exp()), dim=-1)
        kl_loss = torch.mean(kl_loss)

        epoch_estimater_loss += estimater_loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_mi_loss += mi_loss.item()
        epoch_loss += loss.item()

        wandb.log({"val_estimater_loss": estimater_loss})   
        wandb.log({"val_mse_loss": mse_loss})
        wandb.log({"val_kl_loss": kl_loss})
        wandb.log({"val_mi_loss": mi_loss})  
        wandb.log({"val_total_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output_feat[0], cfg, "mel_val_feat", ckpt_time)
                    check_mel(feature[0], output_lip[0], cfg, "mel_val_lip", ckpt_time)
                    if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out_feat[0], cfg, "feat_add_val_feat", ckpt_time)
                        check_feat_add(feature[0], feat_add_out_lip[0], cfg, "feat_add_val_lip", ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output_feat[0], cfg, "mel_val_feat", ckpt_time)
                check_mel(feature[0], output_lip[0], cfg, "mel_val_lip", ckpt_time)
                if cfg.train.use_feat_add:
                    check_feat_add(feature[0], feat_add_out_feat[0], cfg, "feat_add_val_feat", ckpt_time)
                    check_feat_add(feature[0], feat_add_out_lip[0], cfg, "feat_add_val_lip", ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_kl_loss /= iter_cnt
    epoch_mi_loss /= iter_cnt
    epoch_estimater_loss /= iter_cnt
    return epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_kl_loss, epoch_mi_loss, epoch_estimater_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
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

    # 口唇動画か顔かの選択
    lip_or_face = cfg.train.face_or_lip
    if lip_or_face == "face":
        data_root = cfg.train.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    elif lip_or_face == "lip":
        data_root = cfg.train.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    elif lip_or_face == "lip_128128":
        data_root = cfg.train.lip_pre_loaded_path_128128
        mean_std_path = cfg.train.lip_mean_std_path_128128
    elif lip_or_face == "lip_9696":
        data_root = cfg.train.lip_pre_loaded_path_9696
        mean_std_path = cfg.train.lip_mean_std_path_9696
    elif lip_or_face == "lip_9696_time_only":
        data_root = cfg.train.lip_pre_loaded_path_9696_time_only
        mean_std_path = cfg.train.lip_mean_std_path_9696_time_only
    
    data_root = Path(data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")

    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name

    # check point
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / lip_or_face / ckpt_time
    else:
        ckpt_path = ckpt_path / lip_or_face / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / lip_or_face / ckpt_time    
    else:
        save_path = save_path / lip_or_face / current_time
    os.makedirs(save_path, exist_ok=True)

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    # 損失関数
    loss_f = MaskedLoss()
    train_loss_list = []
    train_feat_add_loss_list = []
    train_mse_loss_list = []
    train_kl_loss_list = []
    train_mi_loss_list = []
    train_estimater_loss_list = []
    val_loss_list = []
    val_feat_add_loss_list = []
    val_mse_loss_list = []
    val_ = []
    val_mi_loss_list = []
    val_estimater_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_both"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        mi_estimater = MyCLUBSample(
            x_dim=cfg.model.spk_emb_dim, 
            y_dim=cfg.model.vae_emb_dim,
            hidden_size=cfg.model.mi_hidden_channels
        ).to(device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )
        optimizer_mi = torch.optim.Adam(
            params=mi_estimater.parameters(),
            lr=cfg.train.lr_mi,
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay_mi,
        )

        # scheduler
        # scheduler = CosineLRScheduler(
        #     optimizer, 
        #     t_initial=cfg.train.max_epoch, 
        #     lr_min=cfg.train.lr / 10, 
        #     warmup_t=cfg.train.warmup_t, 
        #     warmup_lr_init=cfg.train.warmup_lr_init, 
        #     warmup_prefix=True,
        # )

        last_epoch = 0

        if cfg.train.check_point_start:
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            mi_estimater.load_state_dict(checkpoint["mi_estimater"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_mi.load_state_dict(checkpoint["optimizer_mi"])
            # scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)
        wandb.watch(mi_estimater, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            # print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")

            # train
            epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_kl_loss, epoch_mi_loss, epoch_estimater_loss = train_one_epoch(
                model=model,
                mi_estimater=mi_estimater,
                train_loader=train_loader,
                optimizer=optimizer,
                optimizer_mi=optimizer_mi,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_feat_add_loss_list.append(epoch_loss_feat_add)
            train_mse_loss_list.append(epoch_mse_loss)
            train_kl_loss_list.append(epoch_kl_loss)
            train_mi_loss_list.append(epoch_mi_loss)
            train_estimater_loss_list.append(epoch_estimater_loss)

            # validation
            epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_kl_loss, epoch_mi_loss, epoch_estimater_loss = val_one_epoch(
                model=model,
                mi_estimater=mi_estimater,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_feat_add_loss_list.append(epoch_loss_feat_add)
            val_mse_loss_list.append(epoch_mse_loss)
            val_.append(epoch_kl_loss)
            val_mi_loss_list.append(epoch_mi_loss)
            val_estimater_loss_list.append(epoch_estimater_loss)

            # scheduler.step(current_epoch)

            # checkpoint
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    mi_estimater=mi_estimater,
                    optimizer=optimizer,
                    optimizer_mi=optimizer_mi,
                    # scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{cfg.model.name}_{current_epoch}.ckpt"),
                )

            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "loss_feat_add")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            save_loss(train_kl_loss_list, val_, save_path, "kl_loss")
            save_loss(train_mi_loss_list, val_mi_loss_list, save_path, "mi_loss")
            save_loss(train_estimater_loss_list, val_estimater_loss_list, save_path, "estimater_loss")

        # save model parameter
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)


if __name__ == "__main__":
    main()