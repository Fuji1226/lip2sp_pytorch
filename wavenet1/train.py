from cProfile import label
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
import matplotlib.pyplot as plt
from librosa.display import specshow

from model.wavenet import WaveNet
from dataset_wavenet import WaveNetDataset, WaveNetTransform
from dataset.dataset_npz import get_datasets
from data_process.mulaw import inv_mulaw_quantize
from data_process.feature import wave2mel


# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
random.seed(7)


def save_checkpoint(model, optimizer, schedular, epoch, ckpt_path):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
                'schedular': schedular.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), 
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                'cuda_random' : torch.cuda.get_rng_state(),
				'epoch': epoch}, ckpt_path)


def make_train_val_loader(cfg, data_root, mean_std_path):
    # パス取得
    data_path = get_datasets(
        data_root=data_root,
        name=cfg.model.name,
    )
    n_samples = len(data_path)
    train_size = int(n_samples * 0.95)
    train_data_path = data_path[:train_size]
    val_data_path = data_path[train_size:]

    # transform
    train_trans = WaveNetTransform(cfg, "train")
    val_trans = WaveNetTransform(cfg, "val")

    # dataset
    train_dataset = WaveNetDataset(
        data_path=train_data_path,
        mean_std_path=mean_std_path,
        transform=train_trans,
        cfg=cfg,
        test=False
    )
    val_dataset = WaveNetDataset(
        data_path=val_data_path,
        mean_std_path=mean_std_path,
        transform=val_trans,
        cfg=cfg,
        test=False
    )

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=False,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return train_loader, val_loader


def make_model(cfg, device):
    model = WaveNet(
        out_channels=cfg.model.out_channels, 
        layers=cfg.model.layers,
        stacks=cfg.model.stacks,
        residual_channels=cfg.model.residual_channels,
        gate_channels=cfg.model.gate_channels,
        skip_out_channels=cfg.model.skip_out_channels,
        kernel_size=cfg.model.kernel_size,
        feature_channels=cfg.model.feature_channels,
        upsample_scales=cfg.model.upsample_scales,
        aux_context_window=cfg.model.aux_context_window,
    ).to(device)
    return model


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()

    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        wav, feature, label = batch
        wav, feature = wav.to(device), feature.to(device)

        out = model(wav, feature)

        loss = loss_f(out, wav)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()

        epoch_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                break

    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    for batch in val_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        wav, feature, label = batch
        wav, feature = wav.to(device), feature.to(device)

        with torch.no_grad():
            out = model(wav, feature)

        loss = loss_f(out, wav)

        epoch_loss += loss.item()
        wandb.log({"val_loss": loss.item()})

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                break

    epoch_loss /= iter_cnt
    return epoch_loss


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    torch.backends.cudnn.benchmark = True

    data_root = cfg.train.data_root
    mean_std_path = cfg.train.mean_std_path
    print("--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")

    # check point
    ckpt_path = os.path.join(cfg.train.ckpt_path, current_time)
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先
    save_path = os.path.join(cfg.train.save_path, current_time)
    os.makedirs(save_path, exist_ok=True)

    # dataloader
    train_loader, val_loader = make_train_val_loader(cfg, data_root, mean_std_path)

    # 損失関数
    loss_f = nn.CrossEntropyLoss()

    # training
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        model = make_model(cfg, device)

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
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
            scheduler.load_state_dict(checkpoint["schedular"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)

        if cfg.train.debug:
            max_epoch = cfg.train.debug_max_epoch
        else:
            max_epoch = cfg.train.max_epoch

        for epoch in range(max_epoch - last_epoch):
            print(f"##### {epoch + last_epoch} #####")
            print(f"learning_rate = {scheduler.get_last_lr()}")

            epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader,
                optimizer=optimizer,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
            )

            if epoch % cfg.train.display_val_loss_step == 0:
                epoch_loss_val = calc_val_loss(
                    model=model,
                    val_loader=val_loader,
                    loss_f=loss_f,
                    device=device,
                    cfg=cfg,
                )

            scheduler.step()

            if epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    schedular=scheduler,
                    epoch=epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.ckpt")
                )       

        torch.save(model.state_dict(), os.path.join(save_path, f"model_{cfg.model.name}.pth"))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(os.path.join(save_path, f"model_{cfg.model.name}.pth"))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__ == "__main__":
    main()