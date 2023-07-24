"""
discriminatorを使う場合のtrain
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import os
from pathlib import Path
import random
from datetime import datetime
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# 自作
from loss import MaskedLoss, AdversarialLoss
from model.discriminator import JCUDiscriminator, SimpleDiscriminator
from train_nar import make_model, make_train_val_loader, save_loss, check_mel, check_feat_add

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def save_checkpoint(model, discriminator, optimizer_gen, optimizer_disc, scheduler_gen, scheduler_disc, epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_gen': optimizer_gen.state_dict(),
        'optimizer_disc': optimizer_disc.state_dict(),
        'scheduler_gen': scheduler_gen.state_dict(),
        'scheduler_disc': scheduler_disc.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def make_discriminator(cfg, device):
    if cfg.model.which_d == "jcu":
        discriminator = JCUDiscriminator(
            in_channels=cfg.model.out_channels,
            out_channels=1,
            use_gc=cfg.train.use_gc,
            emb_in=cfg.train.batch_size,
            dropout=cfg.train.disc_dropout,
        )
    elif cfg.model.which_d == "simple":
        discriminator = SimpleDiscriminator(
            in_channels=cfg.model.out_channels,
            out_channels=1,
        )
    return discriminator.to(device)


def train_one_epoch_with_d(model, discriminator, train_loader, optimizer_gen, optimizer_disc, loss_f, loss_f_adv, device, cfg):
    epoch_loss_gen = 0
    epoch_loss_disc = 0
    epoch_loss_feat_add = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)

    print("iter start")
    model.train()
    discriminator.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        batch_size = lip.shape[0]
        data_cnt += batch_size

        # discriminator
        with torch.no_grad():
            output, feat_add_out = model(lip=lip, data_len=data_len)           
        B, C, T = output.shape
        
        out_f, fmaps_f = discriminator(output)
        out_r, fmaps_r = discriminator(feature)

        loss_disc = loss_f_adv.ls_loss(out_f, out_r, which_loss="d")
        wandb.log({"train_loss_disc": loss_disc.item()})
        epoch_loss_disc += loss_disc.item()
        loss_disc.backward()

        clip_grad_norm_(discriminator.parameters(), cfg.train.max_norm)
        optimizer_disc.step()
        optimizer_disc.zero_grad()

        # generator
        if cfg.train.gen_opt_step:
            output, feat_add_out = model(lip=lip, data_len=data_len)   

            with torch.no_grad():
                out_f, fmaps_f = discriminator(output)
                out_r, fmaps_r = discriminator(feature)

            if cfg.train.multi_task:
                gen_loss_feat_add = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
                wandb.log({"train_loss_gen_feat_add": gen_loss_feat_add.item()})
                epoch_loss_feat_add += gen_loss_feat_add.item()
                gen_loss_feat_add.backward(retain_graph=True)

            gen_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
            loss_g_ls = loss_f_adv.ls_loss(out_f, out_r, which_loss="g")
            loss_g_fm = loss_f_adv.fm_loss(fmaps_f, fmaps_r, data_len=data_len, max_len=T)
            total_loss_gen = cfg.train.mse_weight * gen_loss + cfg.train.ls_weight * loss_g_ls + cfg.train.fm_weight * loss_g_fm
            wandb.log({"train_loss_mse": gen_loss.item()})
            wandb.log({"train_loss_ls": loss_g_ls.item()})
            wandb.log({"train_loss_fm": loss_g_fm.item()})
            wandb.log({"train_loss_gen": total_loss_gen.item()})
            epoch_loss_gen += total_loss_gen.item()
            total_loss_gen.backward()

            clip_grad_norm_(model.parameters(), cfg.train.max_norm)
            optimizer_gen.step()
            optimizer_gen.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output[0], cfg, "mel_train")
                    if cfg.train.multi_task:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train")
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output[0], cfg, "mel_train")
                if cfg.train.multi_task:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train")
    
    epoch_loss_gen /= iter_cnt
    epoch_loss_disc /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss_gen, epoch_loss_disc, epoch_loss_feat_add


def calc_val_loss_with_d(model, discriminator, val_loader, loss_f, loss_f_adv, device, cfg):
    epoch_loss_gen = 0
    epoch_loss_disc = 0
    epoch_loss_feat_add = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()
    discriminator.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        batch_size = lip.shape[0]
        data_cnt += batch_size

        with torch.no_grad():
            output, feat_add_out = model(lip=lip, data_len=data_len)
            out_f, fmaps_f = discriminator(output)   
            out_r, fmaps_r = discriminator(feature)
        
        B, C, T = output.shape

        loss_disc = loss_f_adv.ls_loss(out_f, out_r, which_loss="d")
        wandb.log({"val_loss_disc": loss_disc.item()})
        epoch_loss_disc += loss_disc.item()

        if cfg.train.multi_task:
            gen_loss_feat_add = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            wandb.log({"val_loss_gen_feat_add": gen_loss_feat_add.item()})
            epoch_loss_feat_add += gen_loss_feat_add.item()

        gen_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        loss_g_ls = loss_f_adv.ls_loss(out_f, out_r, which_loss="g")
        loss_g_fm = loss_f_adv.fm_loss(fmaps_f, fmaps_r, data_len=data_len, max_len=T)
        total_loss_gen = cfg.train.mse_weight * gen_loss + cfg.train.ls_weight * loss_g_ls + cfg.train.fm_weight * loss_g_fm
        wandb.log({"val_loss_mse": gen_loss.item()})
        wandb.log({"val_loss_ls": loss_g_ls.item()})
        wandb.log({"val_loss_fm": loss_g_fm.item()})
        wandb.log({"val_loss_gen": total_loss_gen.item()})
        epoch_loss_gen += total_loss_gen.item()
        
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output[0], cfg, "mel_validation")
                    if cfg.train.multi_task:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_validation")
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output[0], cfg, "mel_validation")
                if cfg.train.multi_task:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_validation")
    
    epoch_loss_gen /= iter_cnt
    epoch_loss_disc /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss_gen, epoch_loss_disc, epoch_loss_feat_add


@hydra.main(config_name="config", config_path="conf")
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
    assert lip_or_face == "face" or "lip"
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
    print(f"data_path = {data_root}")
    print(f"mean_std_path = {mean_std_path}")

    # check pointの保存先を指定
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    ckpt_path = ckpt_path / lip_or_face / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = Path(cfg.train.save_path).expanduser()
    save_path = save_path / lip_or_face / current_time
    os.makedirs(save_path, exist_ok=True)
    
    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)
    
    # 損失関数
    loss_f = MaskedLoss()
    loss_f_adv = AdversarialLoss()
    train_loss_list = []
    train_feat_add_loss_list = []
    train_loss_list_d = []
    val_loss_list = []
    val_feat_add_loss_list = []
    val_loss_list_d = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)

        # load parameter
        if cfg.train.load_model:
            if cfg.model.name == "mspec80":
                model_path = cfg.train.model_path_mspec80
            elif cfg.model.name == "world_melfb":
                model_path = cfg.train.model_path_world_melfb
            
            model_path = Path(model_path).expanduser()
            print(f"load {model_path}")
            if model_path.suffix == ".ckpt":
                try:
                    model.load_state_dict(torch.load(str(model_path))['model'])
                except:
                    model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['model'])
            elif model_path.suffix == ".pth":
                try:
                    model.load_state_dict(torch.load(str(model_path)))
                except:
                    model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

        # discriminator
        discriminator = make_discriminator(cfg, device)
        
        # optimizer
        optimizer_gen = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr_gen, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay
        )
        optimizer_disc = torch.optim.Adam(
            discriminator.parameters(), 
            lr=cfg.train.lr_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

        # schedular
        scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_gen, 
            milestones=cfg.train.multi_lr_decay_step_gen,
            gamma=cfg.train.lr_decay_rate_gen      
        )
        scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disc, 
            milestones=cfg.train.multi_lr_decay_step_disc,
            gamma=cfg.train.lr_decay_rate_disc      
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            checkpoint = torch.load(str(checkpoint_path))
            model.load_state_dict(checkpoint["model"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_gen.load_state_dict(checkpoint["optimizer_gen"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            scheduler_gen.load_state_dict(checkpoint["scheduler_gen"])
            scheduler_disc.load_state_dict(checkpoint["scheduler_disc"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)
        wandb.watch(discriminator, **cfg.wandb_conf.watch)

        # training
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = epoch + last_epoch
            print(f"##### {current_epoch} #####")
            print(f"learning_rate_gen = {scheduler_gen.get_last_lr()[0]}")
            print(f"learning_rate_disc = {scheduler_disc.get_last_lr()[0]}")
            
            # training
            train_epoch_loss_gen, train_epoch_loss_disc, train_epoch_loss_feat_add = train_one_epoch_with_d(
                model=model, 
                discriminator=discriminator, 
                train_loader=train_loader, 
                optimizer_gen=optimizer_gen, 
                optimizer_disc=optimizer_disc, 
                loss_f=loss_f, 
                loss_f_adv=loss_f_adv,
                device=device, 
                cfg=cfg,
            )
            train_loss_list.append(train_epoch_loss_gen)
            train_loss_list_d.append(train_epoch_loss_disc)
            train_feat_add_loss_list.append(train_epoch_loss_feat_add)

            # validation
            val_epoch_loss_gen, val_epoch_loss_disc, val_epoch_loss_feat_add = calc_val_loss_with_d(
                model=model, 
                discriminator=discriminator, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                loss_f_adv=loss_f_adv,
                device=device, 
                cfg=cfg,
            )
            val_loss_list.append(val_epoch_loss_gen)
            val_loss_list_d.append(val_epoch_loss_disc)
            val_feat_add_loss_list.append(val_epoch_loss_feat_add)
            
            # 学習率の更新
            scheduler_gen.step()
            scheduler_disc.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    discriminator=discriminator,
                    optimizer_gen=optimizer_gen,
                    optimizer_disc=optimizer_disc,
                    scheduler_gen=scheduler_gen,
                    scheduler_disc=scheduler_disc,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )

            save_loss(train_loss_list, val_loss_list, save_path, "gen_loss")
            save_loss(train_loss_list_d, val_loss_list_d, save_path, "disc_loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "loss_feat_add")

        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)

    wandb.finish()


if __name__=='__main__':
    main()