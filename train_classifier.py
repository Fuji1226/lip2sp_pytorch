from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
from librosa.display import specshow

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from utils import get_path_train, make_train_val_loader, save_loss, count_params, calc_class_balance
from model.classifier import SpeakerClassifier

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


# def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path):
def save_checkpoint(model, optimizer, epoch, ckpt_path):
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


def make_model(cfg, device):
    model = SpeakerClassifier(
        in_channels=80,
        hidden_dim=128,
        n_layers=1,
        bidirectional=True,
        n_speaker=len(cfg.train.speaker),
    ).to(device)
    count_params(model, "model")
    return model


def check_result(target, pred, cfg, filename, epoch, ckpt_time=None):
    pred = F.softmax(pred, dim=-1)
    pred = torch.argmax(pred, dim=-1)
    target = target.to('cpu').detach().numpy().copy()
    pred = pred.to('cpu').detach().numpy().copy()

    acc = np.sum(target == pred) / target.size
    print(acc)

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)

    with open(str(save_path / f"{filename}.txt"), "a") as f:
        f.write(f"epoch = {epoch}\n")
        f.write("--- target ---\n")
        f.write(f"{target}\n")
        f.write("--- pred ---\n")
        f.write(f"{pred}\n")
        f.write(f"accuracy = {acc}\n")
        f.write("\n")


def train_one_epoch(model, train_loader, optimizer, device, cfg, ckpt_time, epoch, class_weight):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        out = model(feature.permute(0, 2, 1))
        loss = F.cross_entropy(out, speaker, weight=class_weight)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        wandb.log({"train_loss": loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_result(speaker, out, cfg, "train", epoch, ckpt_time)
            break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_result(speaker, out, cfg, "train", epoch, ckpt_time)

    return epoch_loss


def val_one_epoch(model, val_loader, device, cfg ,ckpt_time, epoch, class_weight):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start")
    model.train()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        out = model(feature.permute(0, 2, 1))
        loss = F.cross_entropy(out, speaker, weight=class_weight)
        epoch_loss += loss.item()
        wandb.log({"val_loss": loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_result(speaker[0], out[0], cfg, "val", epoch, ckpt_time)
            break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_result(speaker[0], out[0], cfg, "val", epoch, ckpt_time)

    return epoch_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 12
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

    # path
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    # cross entropy weight
    class_weight = calc_class_balance(cfg, data_root, device)

    train_loss_list = []
    val_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        model = make_model(cfg, device)

        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

        last_epoch = 0

        wandb.watch(model, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            # train
            epoch_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                epoch=current_epoch,
                class_weight=class_weight,
            )
            train_loss_list.append(epoch_loss)

            # validation
            epoch_loss = val_one_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                epoch=current_epoch,
                class_weight=class_weight,
            )
            val_loss_list.append(epoch_loss)

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    # scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            save_loss(train_loss_list, val_loss_list, save_path, "loss")

        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)

    wandb.finish()


if __name__ == "__main__":
    main()