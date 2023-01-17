"""
Lipreadingの学習用
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

# 自作
from utils import set_config, get_path_train, make_train_val_loader_lm, count_params, save_loss, check_text
from model.language_model import TransformerLM, RNNLM
from data_process.phoneme_encode import IGNORE_INDEX, SOS_INDEX, EOS_INDEX

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
random.seed(7)


def save_checkpoint(
    model, optimizer, scheduler,
    train_loss_list,
    val_loss_list,
    epoch, ckpt_path):
    torch.save({
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict(),
        "train_loss_list" : train_loss_list,
        "val_loss_list" : val_loss_list,
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch,
    }, ckpt_path)


def make_model(cfg, device):
    # model = TransformerLM(
    #     n_vocab=cfg.model.n_vocab,
    #     d_model=cfg.model.lm_trans_d_model,
    #     n_head=cfg.model.lm_trans_n_head,
    #     n_layers=cfg.model.lm_trans_n_layers,
    # )
    model = RNNLM(
        n_vocab=cfg.model.n_vocab,
        hidden_channels=cfg.model.lm_rnn_hidden_channels,
        n_layers=cfg.model.lm_rnn_n_layers,
        dropout=cfg.model.lm_dropout
    )
    count_params(model, "model")
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, train_loader, dataset, optimizer, device, cfg, ckpt_time, current_epoch):
    epoch_loss = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()
    iter_cnt = 0

    classes_index = dataset.classes_index

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        text, text_len = batch
        text = text.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = text[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = text[:, 1:]    # (B, T)

        output = model(phoneme_index_input, text_len)

        loss = F.cross_entropy(output, phoneme_index_output, ignore_index=IGNORE_INDEX)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        wandb.log({"train_iter_loss": loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_text(phoneme_index_output[0], output[0], current_epoch, cfg, classes_index, "train", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            check_text(phoneme_index_output[0], output[0], current_epoch, cfg, classes_index, "train", current_time, ckpt_time)
            pass
    
    epoch_loss /= iter_cnt
    return epoch_loss


def val_one_epoch(model, val_loader, dataset, device, cfg, ckpt_time, current_epoch):
    epoch_loss = 0
    all_iter = len(val_loader)
    print("iter start")
    model.eval()
    iter_cnt = 0

    classes_index = dataset.classes_index

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        text, text_len = batch
        text = text.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = text[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = text[:, 1:]    # (B, T)

        with torch.no_grad():
            output = model(phoneme_index_input, text_len)

        loss = F.cross_entropy(output, phoneme_index_output, ignore_index=IGNORE_INDEX)

        epoch_loss += loss.item()
        wandb.log({"val_iter_loss": loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_text(phoneme_index_output[0], output[0], current_epoch, cfg, classes_index, "val", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            check_text(phoneme_index_output[0], output[0], current_epoch, cfg, classes_index, "val", current_time, ckpt_time)
            pass
    
    epoch_loss /= iter_cnt
    return epoch_loss


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

    train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"train_data_root = {train_data_root}")
    print(f"val_data_root = {val_data_root}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_lm(cfg, train_data_root, val_data_root)

    train_loss_list = []
    val_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)

        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

        if cfg.train.use_warmup_scheduler:
            scheduler = CosineLRScheduler(
                optimizer, 
                t_initial=cfg.train.max_epoch, 
                lr_min=cfg.train.lr / 10, 
                warmup_t=20, 
                warmup_lr_init=1e-5, 
                warmup_prefix=True,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=cfg.train.lr_decay_exp
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
            train_loss_list = checkpoint["train_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            if cfg.train.use_warmup_scheduler:
                print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")
            else:
                print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            training_method = "tf"

            epoch_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                dataset=train_dataset,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                current_epoch=current_epoch,
            )
            train_loss_list.append(epoch_loss)

            epoch_loss = val_one_epoch(
                model=model,
                val_loader=val_loader,
                dataset=train_dataset,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
                current_epoch=current_epoch,
            )
            val_loss_list.append(epoch_loss)

            if cfg.train.use_warmup_scheduler:
                scheduler.step(current_epoch)
            else:
                scheduler.step()

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_list=train_loss_list,
                    val_loss_list=val_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )

            save_loss(train_loss_list, val_loss_list, save_path, "loss")

        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)

    wandb.finish()


if __name__ == "__main__":
    main()