from omegaconf import OmegaConf
import omegaconf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
import torch
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from utils import (
    set_config,
    fix_random_seed,
    get_save_and_ckpt_path,
    make_train_val_loader_text,
    save_loss,
)
from model.bart import BART

wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss_list,
    val_loss_list,
    epoch,
    ckpt_path,
):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            "random": random.getstate(),
            "np_random": np.random.get_state(), 
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            'cuda_random' : torch.cuda.get_rng_state(),
            'train_loss_list': train_loss_list,
            'val_loss_list': val_loss_list,
            'epoch': epoch,
        }, 
        ckpt_path,
    )


def make_model(
    cfg,
    device,
):
    model = BART(cfg).to(device)
    return model


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    print('training')
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    model.train()
    
    for batch in tqdm(train_loader):
        phoneme_target, phoneme_masked, phoneme_len = batch
        phoneme_target = phoneme_target.to(device)
        phoneme_masked = phoneme_masked.to(device)
        phoneme_len = phoneme_len.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            phoneme_pred = model(
                src_text=phoneme_masked,
                src_text_len=phoneme_len,
                tgt_text=phoneme_target[:, :-1],
                tgt_text_len=phoneme_len - 1,
            )
            phoneme_pred = phoneme_pred.permute(1, 2, 0)
            loss = loss_f(phoneme_pred, phoneme_target[:, 1:])
            epoch_loss += loss.item()
            wandb.log({'train_loss': loss})
            loss = loss / cfg.train.iters_to_accumulate
            
        scaler.scale(loss).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        iter_cnt += 1
        if cfg.train.debug and iter_cnt > cfg.train.debug_iter:
            break
    
    epoch_loss /= iter_cnt
    return epoch_loss


def val_one_epoch(
    model,
    val_loader,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    print('validation')
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    model.eval()
    
    for batch in tqdm(val_loader):
        phoneme_target, phoneme_masked, phoneme_len = batch
        phoneme_target, phoneme_masked, phoneme_len = batch
        phoneme_target = phoneme_target.to(device)
        phoneme_masked = phoneme_masked.to(device)
        phoneme_len = phoneme_len.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                phoneme_pred = model.greedy_search_validation(
                    src_text=phoneme_masked,
                    src_text_len=phoneme_len,
                    iter_limit=phoneme_target.shape[1] - 1,
                    tgt_text=None,
                    tgt_text_len=None,
                )
            phoneme_pred = phoneme_pred.permute(1, 2, 0)
            loss = loss_f(phoneme_pred, phoneme_target[:, 1:])
            epoch_loss += loss.item()
            wandb.log({'val_loss': loss})
            
        iter_cnt += 1
        if cfg.train.debug and iter_cnt > cfg.train.debug_iter:
            break
        if iter_cnt > cfg.model.bart.num_validation:
            break
    
    epoch_loss /= iter_cnt
    return epoch_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
    fix_random_seed(cfg.train.random_seed)
    
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_path, save_path, ckpt_time= get_save_and_ckpt_path(cfg, current_time)
    
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_text(cfg)
    
    loss_f = torch.nn.CrossEntropyLoss(ignore_index=0)
    train_loss_list = []
    val_loss_list = []
    
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        model = make_model(cfg, device)
        
        if cfg.train.which_optim == 'adam':
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )
        elif cfg.train.which_optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )

        if cfg.train.which_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=cfg.train.lr_decay_exp,
            )
        elif cfg.train.which_scheduler == 'warmup':
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=cfg.train.max_epoch,
                lr_min=cfg.train.warmup_lr_min,
                warmup_t=int(cfg.train.max_epoch * cfg.train.warmup_t_rate),
                warmup_lr_init=cfg.train.warmup_lr_init,
                warmup_prefix=True,
            )

        scaler = torch.cuda.amp.GradScaler()
        
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
            scaler.load_state_dict(checkpoint["scaler"])
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
            epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                scaler=scaler,
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            
            epoch_loss = val_one_epoch(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            
            if cfg.train.which_scheduler == 'exp':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == 'warmup':
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
                scheduler.step(epoch)
            
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    val_loss_list=val_loss_list,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{current_epoch}.ckpt"),
                )
            
            save_loss(train_loss_list, val_loss_list, save_path, 'loss')
            
    wandb.finish()
    
    
if __name__ == '__main__':
    main()