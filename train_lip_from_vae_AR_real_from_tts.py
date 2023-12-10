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
from synthesis import generate_for_train_check_lip_from_vqvae_dict

from utils import  make_train_val_loader_stop_token_all, get_path_train, save_loss, check_feat_add, check_mel_default, make_test_loader, check_att, make_test_loader_dict
from model.lip_from_vae import Lip2Sp_VQVAE_TacoAR, Lip_VQENC
from loss import MaskedLoss

from utils import make_test_loader_final, make_train_val_loader_redu4

import psutil

from prob_list import *
from transformers import get_cosine_schedule_with_warmup
from data_process.phoneme_encode import IGNORE_INDEX, SOS_INDEX, EOS_INDEX
from util_from_tts import *

import gc

# wandbへのログイン
wandb.login()

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)

GRAD_OK_EPOCH = 30

def grad_ok_vqvae(model):
    for param in model.vq.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    return model

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

def make_model(cfg, device):
    model = Lip2Sp_VQVAE_TacoAR(
    )
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)

def make_ref_model(cfg, device):
    model = Lip_VQENC()
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, ref_model, train_loader, optimizer, loss_f, device, cfg, training_method, mixing_prob, epoch, ckpt_time, scheduler):
    
    sum_loss = {}
    sum_loss['epoch_output_loss'] = 0
    sum_loss['epoch_vq_loss'] = 0
    sum_loss['epoch_ctc_loss'] = 0
    sum_loss['epoch_stop_token_loss'] = 0
    sum_loss['epoch_ref_loss'] = 0
    
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    optimizer.zero_grad()
    model.train()
  
    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        feature = batch['feature'].to(device)
        data_len = batch['data_len'].to(device)
        lip = batch['lip'].to(device)
        
        text_index = batch['text'].to(device)[:, 1:]
        text_len = batch['text_len'].to(device) -1 
        target_stop_token = batch['stop_token'].to(device)
        
        # output : postnet後の出力
        # dec_output : postnet前の出力
        ref = ref_model(feature, data_len)['vq']
        ref = ref.clone().detach()
        all_output = model(lip, data_len, feature=feature, mode='train', reference=ref)

        output = all_output['output']
        vq_loss = all_output['vq_loss']
        ctc_output = all_output['ctc_output']
        logit = all_output['logit']
        ref_loss = all_output['ref_loss']
        
        B, C, T = output.shape
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T) 
        
        ctc_output = F.log_softmax(ctc_output, dim=-1).permute(1, 0, 2)     # (T, B, C)
        input_lens = torch.full((feature.shape[0],), fill_value=ctc_output.shape[0], dtype=torch.int)
        ctc_loss = F.ctc_loss(ctc_output, text_index, input_lengths=input_lens, target_lengths=text_len, blank=IGNORE_INDEX)

        logit_mask = 1.0 - make_pad_mask_stop_token(data_len, feature.shape[-1]).to(torch.float32)
        logit_mask = logit_mask.to(torch.bool)
        logit = torch.masked_select(logit, logit_mask)
        stop_token = torch.masked_select(target_stop_token, logit_mask)
        stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)
        
        if cfg.train.gradient_accumulation_steps > 1:
            output_loss = output_loss / cfg.train.gradient_accumulation_steps
            vq_loss = vq_loss / cfg.train.gradient_accumulation_steps
            ctc_loss = ctc_loss / cfg.train.gradient_accumulation_steps
            stop_token_loss = stop_token_loss / cfg.train.gradient_accumulation_steps
            ref_loss = ref_loss / cfg.train.gradient_accumulation_steps


        loss = output_loss + vq_loss + ctc_loss + stop_token_loss + ref_loss
        loss.backward()
        sum_loss['epoch_output_loss'] += output_loss.item()
        sum_loss['epoch_vq_loss'] += vq_loss.item()
        sum_loss['epoch_ctc_loss'] += ctc_loss.item()
        sum_loss['epoch_stop_token_loss'] += stop_token_loss.item()
        sum_loss['epoch_ref_loss'] += ref_loss.item()
        
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

        if iter_cnt % cfg.train.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            grad_cnt += 1
        
            if cfg.debug:
                break
            
    
        del feature, data_len, output
        gc.collect()
        torch.cuda.empty_cache()
        plt.clf()
        plt.close()
        
    sum_loss['epoch_output_loss'] /= grad_cnt
    sum_loss['epoch_vq_loss'] /= grad_cnt
    sum_loss['epoch_ctc_loss'] /= grad_cnt
    sum_loss['epoch_stop_token_loss'] /= grad_cnt
    sum_loss['epoch_ref_loss'] /= grad_cnt
    return sum_loss


def calc_val_loss(model, ref_model, val_loader, loss_f, device, cfg, training_method, mixing_prob, ckpt_time):
    sum_loss = {}
    sum_loss['epoch_output_loss'] = 0
    sum_loss['epoch_vq_loss'] = 0
    sum_loss['epoch_ctc_loss'] = 0
    sum_loss['epoch_stop_token_loss'] = 0
    sum_loss['epoch_ref_loss'] = 0
    
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            iter_cnt += 1
            print(f'iter {iter_cnt}/{all_iter}')
            feature = batch['feature'].to(device)
            data_len = batch['data_len'].to(device)
            lip = batch['lip'].to(device)
            
            text_index = batch['text'].to(device)[:, 1:]
            text_len = batch['text_len'].to(device) -1 
            target_stop_token = batch['stop_token'].to(device)
            
            # output : postnet後の出力
            # dec_output : postnet前の出力
            ref = ref_model(feature, data_len)['vq']
            ref = ref.clone().detach()
            all_output = model(lip, data_len, feature=feature, mode='train', reference=ref)
        
            output = all_output['output']
            vq_loss = all_output['vq_loss']
            ctc_output = all_output['ctc_output']
            logit = all_output['logit']
            ref_loss = all_output['ref_loss']
        
            B, C, T = output.shape
        
            output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T) 
            
            ctc_output = F.log_softmax(ctc_output, dim=-1).permute(1, 0, 2)     # (T, B, C)
            input_lens = torch.full((feature.shape[0],), fill_value=ctc_output.shape[0], dtype=torch.int)
            ctc_loss = F.ctc_loss(ctc_output, text_index, input_lengths=input_lens, target_lengths=text_len, blank=IGNORE_INDEX)
                
            logit_mask = 1.0 - make_pad_mask_stop_token(data_len, feature.shape[-1]).to(torch.float32)
            logit_mask = logit_mask.to(torch.bool)
            logit = torch.masked_select(logit, logit_mask)
            stop_token = torch.masked_select(target_stop_token, logit_mask)
            stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)
            
            if cfg.train.gradient_accumulation_steps > 1:
                output_loss = output_loss / cfg.train.gradient_accumulation_steps
                vq_loss = vq_loss / cfg.train.gradient_accumulation_steps
                ctc_loss = ctc_loss / cfg.train.gradient_accumulation_steps
                stop_token_loss = stop_token_loss / cfg.train.gradient_accumulation_steps
                ref_loss = ref_loss / cfg.train.gradient_accumulation_steps
                
            sum_loss['epoch_output_loss'] += output_loss.item()
            sum_loss['epoch_vq_loss'] += vq_loss.item()
            sum_loss['epoch_ctc_loss'] += ctc_loss.item()
            sum_loss['epoch_stop_token_loss'] += stop_token_loss.item()
            sum_loss['epoch_ref_loss'] += ref_loss.item()
            
            if iter_cnt % cfg.train.gradient_accumulation_steps == 0:
                grad_cnt += 1
                if cfg.debug:
                    break
                    
                    
    sum_loss['epoch_output_loss'] /= grad_cnt
    sum_loss['epoch_vq_loss'] /= grad_cnt
    sum_loss['epoch_ctc_loss'] /= grad_cnt
    sum_loss['epoch_stop_token_loss'] /= grad_cnt
    sum_loss['epoch_ref_loss'] /= grad_cnt
    return sum_loss


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
    train_loader, val_loader, train_dataset, val_dataset =  make_train_val_loader_redu4(cfg, data_root, mean_std_path)
    
    test_data_root = [Path(cfg.test.face_pre_loaded_path).expanduser()]
    print(f'test root: {test_data_root}')

    test_loader, test_dataset = make_test_loader_final(cfg, test_data_root, mean_std_path)
    
    # 損失関数
    loss_f = MaskedLoss()
    train_output_loss_list = []
    train_vq_loss_list = []
    train_ctc_loss_list = []
    train_stop_token_loss_list = []
    train_ref_loss_list = []
    
    val_output_loss_list = []
    val_vq_loss_list = []
    val_ctc_loss_list = []
    val_stop_token_loss_list = []
    val_ref_loss_list = []
    
    
    cfg.wandb_conf.setup.name = cfg.tag
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        vq_path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/vq_vae_from_tts/code512_dim256/mspec80_110.ckpt'
        model = load_from_vqvae_ctc_from_taco(model, vq_path)
        
        ref_model = make_ref_model(cfg, device)
        ref_model = load_ref_model(ref_model, vq_path)
        ref_model.eval()
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

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
            
            if epoch >= GRAD_OK_EPOCH:
                grad_ok_vqvae(model)
          
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
            train_sum_loss = train_one_epoch(
                model=model, 
                ref_model=ref_model,
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
            train_output_loss_list.append(train_sum_loss['epoch_output_loss'])
            train_vq_loss_list.append(train_sum_loss['epoch_vq_loss'])
            train_ctc_loss_list.append(train_sum_loss['epoch_ctc_loss'])
            train_stop_token_loss_list.append(train_sum_loss['epoch_stop_token_loss'])
            train_ref_loss_list.append(train_sum_loss['epoch_ref_loss'])


            # validation
            val_sum_loss = calc_val_loss(
                model=model, 
                ref_model=ref_model,
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                training_method=training_method,
                mixing_prob=mixing_prob,
                ckpt_time=ckpt_time,
            )
            val_output_loss_list.append(val_sum_loss['epoch_output_loss'])
            val_vq_loss_list.append(val_sum_loss['epoch_vq_loss'])
            val_ctc_loss_list.append(val_sum_loss['epoch_ctc_loss'])
            val_stop_token_loss_list.append(val_sum_loss['epoch_stop_token_loss'])
            val_ref_loss_list.append(val_sum_loss['epoch_ref_loss'])

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
            
            save_loss(train_output_loss_list, val_output_loss_list, save_path, "output_loss")
            save_loss(train_vq_loss_list, val_vq_loss_list, save_path, "vq_loss")
            save_loss(train_ctc_loss_list, val_ctc_loss_list, save_path, "ctc_loss")
            save_loss(train_stop_token_loss_list, val_stop_token_loss_list, save_path, "stop_token_loss")
            save_loss(train_ref_loss_list, val_ref_loss_list, save_path, "ref_loss")
      

            generate_for_train_check_lip_from_vqvae_dict(
                cfg = cfg,
                model = model,
                test_loader = test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
                epoch=epoch,
                mixing_prob=mixing_prob
            )
        
            if epoch >= GRAD_OK_EPOCH:
                model = model_grad_ok(model)
            
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