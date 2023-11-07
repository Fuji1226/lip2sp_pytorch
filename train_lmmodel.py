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
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from synthesis import generate_for_train_check_taco

from utils import make_train_val_loader_lipread, get_path_train, save_loss, check_feat_add, check_mel_default, make_test_loader, check_att
from model.model_lmmodel import Lip2LM
from loss import MaskedLoss
import torch.nn.functional as F

import psutil

from prob_list import *
from transformers import get_cosine_schedule_with_warmup
from data_process.phoneme_encode import IGNORE_INDEX, SOS_INDEX, EOS_INDEX

from utils import make_train_val_loader_final, make_test_loader_final

import gc
from synthesis_lipread import output_to_text

# wandbへのログイン
wandb.login()

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
    model = Lip2LM(
    )
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, training_method, mixing_prob, epoch, ckpt_time, scheduler):
    loss_dict = {}
    loss_dict['epoch_lm_loss'] = 0
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    optimizer.zero_grad()
    model.train()

    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        # output_to_text(batch['text'][0])
        # breakpoint()
        lip = batch['lip'].to(device)
        data_len = batch['data_len'].to(device)
        text_len = batch['text_len'].to(device)
        text = batch['text'].to(device)
        lip_len = batch['lip_len'].to(device)
        text = text[:, 1:]
        
    
        output_dict = model(lip=lip, prev=text, data_len=data_len, input_len=lip_len)
        
        text_output = output_dict['dec_output']
    
        #loss
        b_size, t_size, _ = text_output.shape
        lm_loss = F.cross_entropy(text_output.reshape(b_size*t_size, -1), text.reshape(-1), ignore_index=0)
        
        if cfg.train.gradient_accumulation_steps > 1:
            loss = lm_loss / cfg.train.gradient_accumulation_steps
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

        loss_dict['epoch_lm_loss'] += loss.item()
        #loss_dict['epoch_ctc_loss'] += loss.item()
        if (iter_cnt) % cfg.train.gradient_accumulation_steps == 0:
            #scaler.step(optimizer)
            #scaler.update()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()
            grad_cnt += 1
            
            break

        del lip, data_len, output_dict, batch
        gc.collect()
        torch.cuda.empty_cache()
        plt.clf()
        plt.close()


    loss_dict['epoch_lm_loss'] /= grad_cnt
    return loss_dict


def calc_val_loss(model, val_loader, loss_f, device, cfg, training_method, mixing_prob, ckpt_time):
    loss_dict = {}
    loss_dict['epoch_lm_loss'] = 0
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            iter_cnt += 1
            print(f'iter {iter_cnt}/{all_iter}')
            
            lip = batch['lip'].to(device)
            data_len = batch['data_len'].to(device)
            text_len = batch['text_len'].to(device)
            text = batch['text'].to(device)
            lip_len = batch['lip_len'].to(device)
            text = text[:, 1:]
            
            
            output_dict = model(lip=lip, prev=text, data_len=data_len, input_len=lip_len)
            
            text_output = output_dict['dec_output']
        
            #loss
            b_size, t_size, _ = text_output.shape
            lm_loss = F.cross_entropy(text_output.reshape(b_size*t_size, -1), text.reshape(-1), ignore_index=0)
            
            
            if cfg.train.gradient_accumulation_steps > 1:
                loss = lm_loss / cfg.train.gradient_accumulation_steps
    
                loss_dict['epoch_lm_loss'] += loss.item()
                
            if (iter_cnt) % cfg.train.gradient_accumulation_steps == 0:
                grad_cnt += 1
                break
            
    loss_dict['epoch_lm_loss'] /= grad_cnt
    return loss_dict

def mixing_prob_controller(mixing_prob, epoch, mixing_prob_change_step):
    """
    mixing_prob_change_stepを超えたらmixing_probを0.01ずつ下げていく
    0.1になったら維持
    """
    if epoch >= mixing_prob_change_step:
        if mixing_prob <= 0.1:
            return mixing_prob
        else:
            mixing_prob -= 0.01
            return mixing_prob
    else:
        return mixing_prob

@hydra.main(config_name="config_all_from_tts_desk", config_path="conf")
def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
        cfg.train.num_workers = 4

    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False
        
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    print(f'tag: {cfg.tag}')
    #breakpoint()
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True


    # 現在時刻を取得
    current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')
    current_time += f'_{cfg.tag}'
    # path
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, train_dataset, val_dataset =  make_train_val_loader_final(cfg, data_root, mean_std_path)
    
    test_data_root = [Path(cfg.test.face_pre_loaded_path).expanduser()]
    print(f'test root: {test_data_root}')

    test_loader, test_dataset = make_test_loader_final(cfg, test_data_root, mean_std_path)
    
    # 損失関数
    loss_f = MaskedLoss()
    train_lm_loss_list = []
    val_lm_loss_list = []

    cfg.wandb_conf.setup.name = cfg.tag
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

        num_warmup_steps = int(len(train_loader)/cfg.train.gradient_accumulation_steps) * 5
        num_training_steps = int(len(train_loader)/cfg.train.gradient_accumulation_steps) * 300

        scheduler = get_cosine_schedule_with_warmup(optimizer, 
            num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

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

        prob_list = mixing_prob_controller_test16(cfg)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
          
            # 学習方法の変更
            if current_epoch < cfg.train.tm_change_step:
                training_method = "tf"  # teacher forcing
            else:
                training_method = "ss"  # scheduled sampling
                
            training_method = cfg.method

            # mixing_probの変更
            if cfg.train.change_mixing_prob:
                if current_epoch >= cfg.train.mp_change_step:
                    if cfg.train.fixed_mixing_prob:
                        mixing_prob = 0.1
                    else:
                        mixing_prob = torch.randint(10, 50, (1,)) / 100     # [0.1, 0.5]でランダム
                        mixing_prob = mixing_prob.item()
                else:
                    mixing_prob = cfg.train.mixing_prob
            else:
                mixing_prob = cfg.train.mixing_prob

            mixing_prob = prob_list[current_epoch-1]

            print(f"training_method : {training_method}")
            print(f"mixing prob: {mixing_prob}")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            # training
            all_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                training_method=training_method,
                mixing_prob=mixing_prob,
                epoch=current_epoch,
                ckpt_time=ckpt_time,
                scheduler=scheduler
            )
            train_lm_loss_list.append(all_loss['epoch_lm_loss'])

            # validation
            all_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                training_method=training_method,
                mixing_prob=mixing_prob,
                ckpt_time=ckpt_time,
            )
            val_lm_loss_list.append(all_loss['epoch_lm_loss'])

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            save_loss(train_lm_loss_list, val_lm_loss_list, save_path, "lm_loss")

            mem = psutil.virtual_memory() 
            print(f'cpu usage: {mem.percent}')
            plt.clf()
            plt.close()

        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()