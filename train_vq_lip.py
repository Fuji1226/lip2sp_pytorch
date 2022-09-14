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
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from train_default import make_train_val_loader, check_feat_add, save_loss, get_path
from train_nar import check_mel
from train_vq_audio import make_model
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(model, optimizer, epoch, ckpt_path):
# def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def check_idx(target, output, cfg, filename, epoch, ckpt_time=None):
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)

    with open(str(save_path / f"{filename}.txt"), mode="a") as f:
        f.write(f"\n--- epoch {epoch} ---\n")
        f.write("answer\n")
        f.write(f"{target}\n")
        f.write("\npredict\n")
        f.write(f"{output}\n")


def train_one_epoch(model, train_loader, optimizer, loss_f, loss_f_mse, device, cfg, ckpt_time, epoch):
    epoch_loss = 0
    epoch_loss_mse = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        # 事前学習済みモデルのembed_idxをラベルとする
        with torch.no_grad():
            _, _, _, _, _, embed_idx_target, _, _, _ = model(feature=feature, feature_ref=feature_ref, data_len=data_len)

        output, feat_add_out, phoneme, spk_emb, quantize, embed_idx, vq_loss, enc_output, idx_pred = model(lip=lip, feature_ref=feature_ref, data_len=data_len)
        B, C, T = output.shape

        loss = loss_f(enc_output.permute(0, 2, 1), embed_idx_target)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        wandb.log({"train_loss": loss})

        mse_loss = loss_f_mse.mse_loss(output, feature, data_len, max_len=T)
        epoch_loss_mse += mse_loss.item()
        wandb.log({"train_mse_loss": mse_loss})

        if cfg.train.use_feat_add:
            loss_feat_add = loss_f_mse.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"train_loss_feat_add": loss_feat_add})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    print(f"embed_idx_target = {type(embed_idx_target)}, {embed_idx_target.shape}")
                    print(f"embed_idx = {type(idx_pred)}, {idx_pred.shape}")
                    check_idx(embed_idx_target[0], idx_pred[0], cfg, "idx", epoch, ckpt_time)
                    check_mel(feature[0], output[0], cfg, "mel_train", ckpt_time)
                    if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train", ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_idx(embed_idx_target[0], idx_pred[0], cfg, "idx", epoch, ckpt_time)
                check_mel(feature[0], output[0], cfg, "mel_train", ckpt_time)
                if cfg.train.use_feat_add:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train", ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_mse /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss, epoch_loss_mse, epoch_loss_feat_add

def val_one_epoch(model, val_loader, loss_f, loss_f_mse, device, cfg, ckpt_time, epoch):
    epoch_loss = 0
    epoch_loss_mse = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start") 
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        with torch.no_grad():
            _, _, _, _, _, embed_idx_target, _, _, _ = model(feature=feature, feature_ref=feature_ref, data_len=data_len)
            output, feat_add_out, phoneme, spk_emb, quantize, embed_idx, vq_loss, enc_output, idx_pred = model(lip=lip, feature_ref=feature_ref, data_len=data_len)
        B, C, T = output.shape

        loss = loss_f(enc_output.permute(0, 2, 1), embed_idx_target)
        epoch_loss += loss.item()
        wandb.log({"val_loss": loss})

        mse_loss = loss_f_mse.mse_loss(output, feature, data_len, max_len=T)
        epoch_loss_mse += mse_loss.item()
        wandb.log({"val_mse_loss": mse_loss})

        if cfg.train.use_feat_add:
            loss_feat_add = loss_f_mse.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"val_loss_feat_add": loss_feat_add})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_idx(embed_idx_target[0], idx_pred[0], cfg, "idx", epoch, ckpt_time)
                    check_mel(feature[0], output[0], cfg, "mel_val", ckpt_time)
                    if cfg.train.use_feat_add:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_val", ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_idx(embed_idx_target[0], idx_pred[0], cfg, "idx", epoch, ckpt_time)
                check_mel(feature[0], output[0], cfg, "mel_val", ckpt_time)
                if cfg.train.use_feat_add:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_val", ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_mse /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss, epoch_loss_mse, epoch_loss_feat_add


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
        cfg.train.num_workers = 4

    cfg.train.n_speaker = len(cfg.train.speaker)

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

    loss_f = nn.CrossEntropyLoss()
    loss_f_mse = MaskedLoss()
    train_loss_list = []
    train_mse_loss_list = []
    train_feat_add_loss_list = []
    val_loss_list = []
    val_mse_loss_list = []
    val_feat_add_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_lip"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        model_path = Path(cfg.train.model_path).expanduser()

        # if model_path.suffix == ".ckpt":
        #     try:
        #         model.load_state_dict(torch.load(str(model_path))['model'])
        #     except:
        #         model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['model'])
        # elif model_path.suffix == ".pth":
        #     try:
        #         model.load_state_dict(torch.load(str(model_path)))
        #     except:
        #         model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
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
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["scheduler"])
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
            # print(f"learning_rate = {scheduler_mi.get_last_lr()[0]}")
            # print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")

            # train
            epoch_loss, epoch_loss_mse, epoch_loss_feat_add = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_f=loss_f,
                loss_f_mse=loss_f_mse,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                epoch=current_epoch,
            )
            train_loss_list.append(epoch_loss)
            train_mse_loss_list.append(epoch_loss_mse)
            train_feat_add_loss_list.append(epoch_loss_feat_add)

            # validation
            epoch_loss, epoch_loss_mse, epoch_loss_feat_add = val_one_epoch(
                model=model,
                val_loader=val_loader,
                loss_f=loss_f,
                loss_f_mse=loss_f_mse,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                epoch=current_epoch,
            )
            val_loss_list.append(epoch_loss)
            val_mse_loss_list.append(epoch_loss_mse)
            val_feat_add_loss_list.append(epoch_loss_feat_add)

            # scheduler.step(current_epoch)

            # checkpoint
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    # scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{cfg.model.name}_{current_epoch}.ckpt"),
                )

            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "feat_add_loss")

        # save model
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)


if __name__ == "__main__":
    main()