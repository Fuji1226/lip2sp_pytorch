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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import make_train_val_loader, save_loss, get_path_train, check_mel_nar
from train_ae_audio import make_model
from loss import MaskedLoss
from model.discriminator import AEDiscriminator

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


# def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path):
def save_checkpoint(lip_enc, discriminator, optimizer, optimizer_disc, epoch, ckpt_path):
	torch.save({
        'lip_enc': lip_enc.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_disc': optimizer_disc.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def train_one_epoch(vcnet, lip_enc, discriminator, train_loader, optimizer, optimizer_disc, loss_f, device, cfg, ckpt_time, epoch):
    epoch_loss_enc = 0
    epoch_loss_mel = 0
    epoch_loss_feat_add = 0
    epoch_loss_disc = 0
    epoch_loss_gen = 0
    epoch_loss_total = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    vcnet.train()
    lip_enc.train()
    discriminator.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        # 事前学習済みモデルのembed_idxをラベルとする
        with torch.no_grad():
            _, _, _, _, enc_output_target, _, _ = vcnet(feature=feature, feature_ref=feature, data_len=data_len)

        ### update discriminator ###
        with torch.no_grad():
            lip_enc_out = lip_enc(lip=lip, data_len=data_len)
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(lip_enc_out=lip_enc_out, feature_ref=feature, data_len=data_len)
        
        real = discriminator(enc_output_target)
        fake = discriminator(lip_enc_out)

        disc_loss = F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake))
        disc_loss.backward()
        clip_grad_norm_(discriminator.parameters(), cfg.train.max_norm)
        optimizer_disc.step()
        optimizer_disc.zero_grad()

        epoch_loss_disc += disc_loss.item()
        wandb.log({"train_loss_disc": disc_loss})

        ### update generator ###
        lip_enc_out = lip_enc(lip=lip, data_len=data_len)
        with torch.no_grad():
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(lip_enc_out=lip_enc_out, feature_ref=feature, data_len=data_len)
            fake = discriminator(lip_enc_out)

        gen_loss = F.mse_loss(fake, torch.ones_like(fake))
        enc_loss = loss_f.mse_loss(
            lip_enc_out.permute(0, 2, 1), enc_output_target.permute(0, 2, 1), 
            data_len = torch.div(data_len, 2).to(dtype=torch.int), max_len=lip_enc_out.shape[1]
        )
        loss = enc_loss + cfg.train.gan_weight * gen_loss
        loss.backward()
        clip_grad_norm_(vcnet.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss_enc += enc_loss.item()
        epoch_loss_gen += gen_loss.item()
        epoch_loss_total += loss.item()
        wandb.log({"train_loss_enc": enc_loss})
        wandb.log({"train_loss_gen": gen_loss})
        wandb.log({"train_loss_total": loss})

        mel_loss = loss_f.mse_loss(output, feature, data_len, max_len=output.shape[-1])
        epoch_loss_mel += mel_loss.item()
        wandb.log({"train_loss_mel": mel_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)

    epoch_loss_enc /= iter_cnt
    epoch_loss_mel /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_loss_disc /= iter_cnt
    epoch_loss_gen /= iter_cnt
    epoch_loss_total /= iter_cnt
    return epoch_loss_enc, epoch_loss_mel, epoch_loss_feat_add, epoch_loss_disc, epoch_loss_gen, epoch_loss_total

def val_one_epoch(vcnet, lip_enc, discriminator, val_loader, loss_f, device, cfg, ckpt_time, epoch):
    epoch_loss_enc = 0
    epoch_loss_mel = 0
    epoch_loss_feat_add = 0
    epoch_loss_disc = 0
    epoch_loss_gen = 0
    epoch_loss_total = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start") 
    vcnet.eval()
    lip_enc.eval()
    discriminator.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        with torch.no_grad():
            _, _, _, _, enc_output_target, _, _ = vcnet(feature=feature, feature_ref=feature, data_len=data_len)
            lip_enc_out = lip_enc(lip=lip, data_len=data_len)
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(lip_enc_out=lip_enc_out, feature_ref=feature, data_len=data_len)
            real = discriminator(enc_output_target)
            fake = discriminator(lip_enc_out)

        disc_loss = F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake))
        enc_loss = loss_f.mse_loss(
            lip_enc_out.permute(0, 2, 1), enc_output_target.permute(0, 2, 1), 
            data_len = torch.div(data_len, 2).to(dtype=torch.int), max_len=lip_enc_out.shape[1]
        )
        gen_loss = F.mse_loss(fake, torch.ones_like(fake))
        loss = enc_loss + cfg.train.gan_weight * gen_loss

        epoch_loss_disc += disc_loss.item()
        epoch_loss_enc += enc_loss.item()
        epoch_loss_gen += gen_loss.item()
        epoch_loss_total += loss.item()
        wandb.log({"train_loss_disc": disc_loss})
        wandb.log({"train_loss_enc": enc_loss})
        wandb.log({"train_loss_gen": gen_loss})
        wandb.log({"train_loss_total": loss})

        mel_loss = loss_f.mse_loss(output, feature, data_len, max_len=output.shape[-1])
        epoch_loss_mel += mel_loss.item()
        wandb.log({"val_loss_mel": mel_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_val", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_val", current_time, ckpt_time)

    epoch_loss_enc /= iter_cnt
    epoch_loss_mel /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_loss_disc /= iter_cnt
    epoch_loss_gen /= iter_cnt
    epoch_loss_total /= iter_cnt
    return epoch_loss_enc, epoch_loss_mel, epoch_loss_feat_add, epoch_loss_disc, epoch_loss_gen, epoch_loss_total


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
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    loss_f = MaskedLoss()
    train_enc_loss_list = []
    train_mel_loss_list = []
    train_feat_add_loss_list = []
    train_disc_loss_list = []
    train_gen_loss_list = []
    train_total_loss_list = []
    val_enc_loss_list = []
    val_mel_loss_list = []
    val_feat_add_loss_list = []
    val_disc_loss_list = []
    val_gen_loss_list = []
    val_total_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_lip"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        vcnet, lip_enc = make_model(cfg, device)
        model_path = Path(cfg.train.model_path).expanduser()

        # if model_path.suffix == ".ckpt":
        #     try:
        #         vcnet.load_state_dict(torch.load(str(model_path))['vcnet'])
        #     except:
        #         vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['vcnet'])
        # elif model_path.suffix == ".pth":
        #     try:
        #         vcnet.load_state_dict(torch.load(str(model_path)))
        #     except:
        #         vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

        discriminator = AEDiscriminator(cfg.model.ae_emb_dim).to(device)

        # optimizer
        optimizer = torch.optim.Adam(
            params=lip_enc.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )
        optimizer_disc = torch.optim.Adam(
            params=discriminator.parameters(),
            lr=cfg.train.lr_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
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
            lip_enc.load_state_dict(checkpoint["lip_enc"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            # scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
        
        wandb.watch(vcnet, **cfg.wandb_conf.watch)
        wandb.watch(lip_enc, **cfg.wandb_conf.watch)
        wandb.watch(discriminator, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            # print(f"learning_rate = {scheduler_mi.get_last_lr()[0]}")
            # print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")

            # train
            epoch_loss_enc, epoch_loss_mel, epoch_loss_feat_add, epoch_loss_disc, epoch_loss_gen, epoch_loss_total = train_one_epoch(
                vcnet=vcnet,
                lip_enc=lip_enc,
                discriminator=discriminator,
                train_loader=train_loader,
                optimizer=optimizer,
                optimizer_disc=optimizer_disc,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                epoch=current_epoch,
            )
            train_enc_loss_list.append(epoch_loss_enc)
            train_mel_loss_list.append(epoch_loss_mel)
            train_feat_add_loss_list.append(epoch_loss_feat_add)
            train_disc_loss_list.append(epoch_loss_disc)
            train_gen_loss_list.append(epoch_loss_gen)
            train_total_loss_list.append(epoch_loss_total)

            # validation
            epoch_loss_enc, epoch_loss_mel, epoch_loss_feat_add, epoch_loss_disc, epoch_loss_gen, epoch_loss_total = val_one_epoch(
                vcnet=vcnet,
                lip_enc=lip_enc,
                discriminator=discriminator,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                epoch=current_epoch,
            )
            val_enc_loss_list.append(epoch_loss_enc)
            val_mel_loss_list.append(epoch_loss_mel)
            val_feat_add_loss_list.append(epoch_loss_feat_add)
            val_disc_loss_list.append(epoch_loss_disc)
            val_gen_loss_list.append(epoch_loss_gen)
            val_total_loss_list.append(epoch_loss_total)

            # scheduler.step(current_epoch)

            # checkpoint
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    lip_enc=lip_enc,
                    discriminator=discriminator,
                    optimizer=optimizer,
                    optimizer_disc=optimizer_disc,
                    # scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{cfg.model.name}_{current_epoch}.ckpt"),
                )

            # save loss
            save_loss(train_enc_loss_list, val_enc_loss_list, save_path, "enc_loss")
            save_loss(train_mel_loss_list, val_mel_loss_list, save_path, "mel_loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "feat_add_loss")
            save_loss(train_disc_loss_list, val_disc_loss_list, save_path, "disc_loss")
            save_loss(train_gen_loss_list, val_gen_loss_list, save_path, "gen_loss")
            save_loss(train_total_loss_list, val_total_loss_list, save_path, "total_loss")

        # save model
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(lip_enc.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)


if __name__ == "__main__":
    main()