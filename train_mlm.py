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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from synthesis import generate_for_train_check_vqvae_dict

from utils import  make_train_val_loader_stop_token_all, get_path_train, save_loss, check_feat_add, check_mel_default, make_test_loader, check_att, make_test_loader_dict
from model.vq_vae import VQVAE_Content_ResTC
from loss import MaskedLoss

from dataset.dataset_npz_mlm_final import make_train_val_loader_mlm

import psutil

from prob_list import *
from transformers import get_cosine_schedule_with_warmup
from data_process.phoneme_encode import IGNORE_INDEX, SOS_INDEX, EOS_INDEX
from util_from_tts import *

from model.mlm import MLMTrainer
import gc

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
 
def make_pad_mask_stop_token(lengths, max_len):
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = mask.to(device=device)  # マスクをデバイスに送る

    return mask

def make_mlm(cfg, device, weight_path):
    
    pretrained_embedding_weights = np.load(weight_path)
    pretrained_embedding_weights = torch.tensor(pretrained_embedding_weights)
    
    n_code = pretrained_embedding_weights.shape[0]
    n_dim = pretrained_embedding_weights.shape[1]
    
    trainer = MLMTrainer(n_code=n_code, d_model=n_dim)
    
    remaining_weights = torch.randn(2, n_dim)
    combined_weights = torch.cat([pretrained_embedding_weights, remaining_weights], dim=0)
    
    trainer.encoder.emb.weight.data.copy_(combined_weights)
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        trainer = torch.nn.DataParallel(trainer)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return trainer.to(device)


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, scheduler):
    
    sum_loss = {}
    sum_loss['epoch_mlm_loss'] = 0
    sum_loss['epoch_vq_loss'] = 0
    
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    optimizer.zero_grad()
    model.train()
  
    for batch in train_loader:
        vq_idx = batch['encoding']
        data_len = batch['data_len']
        
        mlm_loss = model(vq_idx, data_len, device)
        mlm_loss.backward()
        sum_loss['epoch_mlm_loss'] += mlm_loss.item()
        
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

       
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        grad_cnt += 1
        
        if cfg.debug:
            break
            
        gc.collect()
        torch.cuda.empty_cache()
        plt.clf()
        plt.close()
        
    sum_loss['epoch_mlm_loss'] /= grad_cnt
    return sum_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg):
    sum_loss = {}
    sum_loss['epoch_mlm_loss'] = 0

    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            iter_cnt += 1
            vq_idx = batch['encoding']
            data_len = batch['data_len']
            
            mlm_loss = model(vq_idx, data_len, device)
            sum_loss['epoch_mlm_loss'] += mlm_loss.item()
        
            grad_cnt += 1
            if cfg.debug:
                break
                    
                    
    sum_loss['epoch_mlm_loss'] /= grad_cnt
    return sum_loss


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
    train_loader, val_loader, train_dataset, val_dataset =  make_train_val_loader_mlm(cfg, data_root, mean_std_path)
    
    test_data_root = [Path(cfg.test.face_pre_loaded_path).expanduser()]
    print(f'test root: {test_data_root}')

    
    # 損失関数
    loss_f = MaskedLoss()
    train_mlm_loss_list = []
    
    val_mlm_loss_list = []
    
    cfg.wandb_conf.setup.name = cfg.tag
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    

    # 2. 事前に学習された埋め込み層の重みを読み込む
    path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/vq_idx/emb_wieghts.npy'
    
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
    # 2. 事前に学習された埋め込み層の重みを読み込む
        path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/vq_idx/emb_wieghts.npy'
        model = make_mlm(cfg, device, path)
            

        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )
        # scheduler
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer=optimizer,
        #     milestones=cfg.train.multi_lr_decay_step,
        #     gamma=cfg.train.lr_decay_rate,
        # )

        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=100000)
        num_warmup_steps = len(train_loader) * 5

        num_training_steps = len(train_loader) * 300

        scheduler = get_cosine_schedule_with_warmup(optimizer, 
            num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # scheduler = CosineLRScheduler(
        #     optimizer, 
        #     t_initial=cfg.train.max_epoch, 
        #     lr_min=cfg.train.lr / 10, 
        #     warmup_t=cfg.train.warmup_t, 
        #     warmup_lr_init=cfg.train.warmup_lr_init, 
        #     warmup_prefix=True,
        # )

        last_epoch = 0

        wandb.watch(model, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
          
            # 学習方法の変更
            if current_epoch < cfg.train.tm_change_step:
                training_method = "tf"  # teacher forcing
            else:
                training_method = "ss"  # scheduled sampling
                
            training_method = cfg.method

            # training
            train_sum_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                scheduler=scheduler
            )
            train_mlm_loss_list.append(train_sum_loss['epoch_mlm_loss'])


            # validation
            val_sum_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
            )
            val_mlm_loss_list.append(val_sum_loss['epoch_mlm_loss'])

            #scheduler.step()

            # check point
            if current_epoch % 10 == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            save_loss(train_mlm_loss_list, val_mlm_loss_list, save_path, "output_loss")

        
            
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