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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from model.model_ae import Lip2SP_AE
from loss import MaskedLoss
from train_default import make_train_val_loader, check_feat_add, save_loss, get_path
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
    model = Lip2SP_AE(
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
        ae_emb_dim=cfg.model.ae_emb_dim,
        spk_emb_dim=cfg.model.spk_emb_dim,
        time_reduction=cfg.train.time_reduction,
        time_reduction_rate=cfg.train.time_reduction_rate,
        n_speaker=len(cfg.train.speaker),
        norm_type_lip=cfg.model.norm_type_lip,
        norm_type_audio=cfg.model.norm_type_audio,
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
    epoch_mi_loss = 0
    epoch_classifier_loss = 0
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
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        ### update mi estimater ###
        with torch.no_grad():
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class = model(feature=feature, feat_add=feat_add, feature_ref=feature, data_len=data_len)
        
        estimater_loss = - mi_estimater.loglikeli(spk_emb, enc_output)
        estimater_loss.backward()
        clip_grad_norm_(mi_estimater.parameters(), cfg.train.max_norm)
        optimizer_mi.step()
        optimizer_mi.zero_grad()

        epoch_estimater_loss += estimater_loss.item()
        wandb.log({"train_estimater_loss": estimater_loss})

        ### update model ###
        output, feat_add_out, phoneme, spk_emb, enc_output, spk_class = model(feature=feature, feat_add=feat_add, feature_ref=feature, data_len=data_len)
        B, C, T = output.shape

        loss = 0
        if cfg.train.use_feat_add:
            loss_feat_add = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            loss += loss_feat_add
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"train_loss_feat_add": loss_feat_add})

        mse_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)

        with torch.no_grad():
            mi_loss = mi_estimater(spk_emb, enc_output)

        classifier_loss = F.cross_entropy(spk_class, speaker)

        loss = loss + mse_loss + cfg.train.mi_weight * mi_loss + cfg.train.classifier_weight * classifier_loss
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_mse_loss += mse_loss.item()
        epoch_mi_loss += mi_loss.item()
        epoch_loss += loss.item()
        epoch_classifier_loss += classifier_loss.item()
        wandb.log({"train_mse_loss": mse_loss})
        wandb.log({"train_mi_loss": mi_loss})
        wandb.log({"train_classifier_loss": classifier_loss})
        wandb.log({"train_total_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output[0], cfg, "mel_train", ckpt_time)
                    if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train", ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output[0], cfg, "mel_train", ckpt_time)
                if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train", ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_mi_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    epoch_estimater_loss /= iter_cnt
    return epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss


def val_one_epoch(model, mi_estimater, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    epoch_mse_loss = 0
    epoch_mi_loss = 0
    epoch_classifier_loss = 0
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
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        with torch.no_grad():
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class = model(feature=feature, feature_ref=feature, data_len=data_len)
            estimater_loss = - mi_estimater.loglikeli(spk_emb, enc_output)
            mi_loss = mi_estimater(spk_emb, enc_output)
        B, C, T = output.shape

        loss = 0
        if cfg.train.use_feat_add:
            loss_feat_add = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            loss += loss_feat_add
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"val_loss_feat_add": loss_feat_add})

        mse_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        classifier_loss = F.cross_entropy(spk_class, speaker)
        loss = loss + mse_loss + cfg.train.mi_weight * mi_loss + cfg.train.classifier_weight * classifier_loss

        epoch_estimater_loss += estimater_loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_mi_loss += mi_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        epoch_loss += loss.item()

        wandb.log({"val_estimater_loss": estimater_loss})   
        wandb.log({"val_mse_loss": mse_loss})
        wandb.log({"val_mi_loss": mi_loss})  
        wandb.log({"val_classifier_loss": classifier_loss})
        wandb.log({"val_total_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output[0], cfg, "mel_val", ckpt_time)
                    if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_val", ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output[0], cfg, "mel_val", ckpt_time)
                if cfg.train.use_feat_add:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_val", ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_mi_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    epoch_estimater_loss /= iter_cnt
    return epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss

@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
        cfg.train.num_workers = 4
        
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
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path(cfg)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    # 損失関数
    loss_f = MaskedLoss()
    train_loss_list = []
    train_feat_add_loss_list = []
    train_mse_loss_list = []
    train_mi_loss_list = []
    train_estimater_loss_list = []
    train_classifier_loss_list = []
    val_loss_list = []
    val_feat_add_loss_list = []
    val_mse_loss_list = []
    val_mi_loss_list = []
    val_estimater_loss_list = []
    val_classifier_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_audio"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        mi_estimater = MyCLUBSample(
            x_dim=cfg.model.spk_emb_dim, 
            y_dim=cfg.model.ae_emb_dim,
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
            epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss = train_one_epoch(
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
            train_mi_loss_list.append(epoch_mi_loss)
            train_estimater_loss_list.append(epoch_estimater_loss)
            train_classifier_loss_list.append(epoch_classifier_loss)

            # validation
            epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss = val_one_epoch(
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
            val_mi_loss_list.append(epoch_mi_loss)
            val_estimater_loss_list.append(epoch_estimater_loss)
            val_classifier_loss_list.append(epoch_classifier_loss)

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
            save_loss(train_mi_loss_list, val_mi_loss_list, save_path, "mi_loss")
            save_loss(train_estimater_loss_list, val_estimater_loss_list, save_path, "estimater_loss")
            save_loss(train_classifier_loss_list, val_classifier_loss_list, save_path, "classifier_loss")

        # save model parameter
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)


if __name__ == "__main__":
    main()