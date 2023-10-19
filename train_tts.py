from omegaconf import DictConfig, OmegaConf
import gc

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
import torch.nn.functional as F

from model.tts_taco import TTSTacotron
from loss import MaskedLossTTS

from utils import get_path_tts_train, make_train_val_loader_tts_multi, make_pad_mask_tts, check_mel_default, check_attention_weight, save_loss, make_test_loader_tts, make_test_loader_tts_re
from synthesis_tts import generate_for_tts

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)

def save_checkpoint(
    model, optimizer, scheduler,
    train_loss_list,
    train_output_loss_list,
    train_dec_output_loss_list,
    train_stop_token_loss_list,
    val_loss_list,
    val_output_loss_list,
    val_dec_output_loss_list,
    val_stop_token_loss_list,
    epoch, ckpt_path):

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        "train_loss_list" : train_loss_list,
        "train_output_loss_list" : train_output_loss_list,
        "train_dec_output_loss_list" : train_dec_output_loss_list,
        "train_stop_token_loss_list" : train_stop_token_loss_list,
        "val_loss_list" : val_loss_list,
        "val_output_loss_list" : val_output_loss_list,
        "val_dec_output_loss_list" : val_dec_output_loss_list,
        "val_stop_token_loss_list" : val_stop_token_loss_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    model = TTSTacotron(cfg)
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)

def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg):
    epoch_loss = 0
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_stop_token_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("start training")
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, feature, text, stop_token, feature_len, text_len, filename = batch
        
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)

        dec_output, output, logit, att_w = model(text, text_len, feature_target=feature)
        dec_output_loss = loss_f.mse_loss(dec_output, feature, feature_len, feature.shape[-1])
        output_loss = loss_f.mse_loss(output, feature, feature_len, feature.shape[-1])

        logit_mask = 1.0 - make_pad_mask_tts(feature_len, feature.shape[-1]).to(torch.float32).squeeze(1)
        logit_mask = logit_mask.to(torch.bool)
        logit = torch.masked_select(logit, logit_mask)
        stop_token = torch.masked_select(stop_token, logit_mask)
        stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)

        total_loss = dec_output_loss + output_loss + stop_token_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += total_loss.item()
        epoch_output_loss += output_loss.item()
        epoch_dec_output_loss += dec_output_loss.item()
        epoch_stop_token_loss += stop_token_loss.item()

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time)
                check_attention_weight(att_w[0], cfg, "att_w_train", current_time)
                

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time)
            check_attention_weight(att_w[0], cfg, "att_w_train", current_time)

    epoch_loss /= iter_cnt
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    epoch_stop_token_loss /= iter_cnt
    return epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_stop_token_loss


def val_one_epoch(model, val_loader, loss_f, device, cfg):
    epoch_loss = 0
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_stop_token_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("start validation")
    model.eval()

    for batch in val_loader:
        wav, feature, text, stop_token, feature_len, text_len, filename = batch
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)


        with torch.no_grad():
            dec_output, output, logit, att_w = model(text, text_len, feature_target=feature)

        dec_output_loss = loss_f.mse_loss(dec_output, feature, feature_len, feature.shape[-1])
        output_loss = loss_f.mse_loss(output, feature, feature_len, feature.shape[-1])

        logit_mask = 1.0 - make_pad_mask_tts(feature_len, feature.shape[-1]).to(torch.float32).squeeze(1)
        logit_mask = logit_mask.to(torch.bool)
        logit = torch.masked_select(logit, logit_mask)
        stop_token = torch.masked_select(stop_token, logit_mask)
        stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)

        total_loss = dec_output_loss + output_loss + stop_token_loss

        epoch_loss += total_loss.item()
        epoch_output_loss += output_loss.item()
        epoch_dec_output_loss += dec_output_loss.item()
        epoch_stop_token_loss += stop_token_loss.item()

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_val", current_time)
                check_attention_weight(att_w[0], cfg, "att_w_val", current_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_val", current_time)
            check_attention_weight(att_w[0], cfg, "att_w_val", current_time)

    epoch_loss /= iter_cnt
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    epoch_stop_token_loss /= iter_cnt
    return epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_stop_token_loss



@hydra.main(config_name="config_tts_desk", config_path="conf")
def main(cfg):
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
    data_root, save_path, ckpt_path = get_path_tts_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")
    
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_tts_multi(cfg, data_root)
    
    test_root = Path(cfg.test.tts_pre_loaded_path).expanduser()
    test_loader, test_dataset = make_test_loader_tts(cfg, test_root, data_root)
    loss_f = MaskedLossTTS()    

    train_loss_list = []
    train_output_loss_list = []
    train_dec_output_loss_list = []
    train_stop_token_loss_list = []
    val_loss_list = []
    val_output_loss_list = []
    val_dec_output_loss_list = []
    val_stop_token_loss_list = []
    
    
    #model
    model = make_model(cfg, device)
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.train.lr, 
        betas=(cfg.train.beta_1, cfg.train.beta_2),
        weight_decay=cfg.train.weight_decay,    
    )

    # scheduler    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.train.lr_decay_exp
    )

    last_epoch = 0
    for epoch in range(cfg.train.max_epoch - last_epoch):
        current_epoch = 1 + epoch + last_epoch
        print(f"##### {current_epoch} #####")
        print(f"learning_rate = {scheduler.get_last_lr()[0]}")
        
        epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_stop_token_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_f=loss_f,
            device=device,
            cfg=cfg
        )
        train_loss_list.append(epoch_loss)
        train_output_loss_list.append(epoch_output_loss)
        train_dec_output_loss_list.append(epoch_dec_output_loss)
        train_stop_token_loss_list.append(epoch_stop_token_loss)

        epoch_loss, epoch_output_loss, epoch_dec_output_loss, epoch_stop_token_loss = val_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_f=loss_f,
            device=device,
            cfg=cfg,
        )
        val_loss_list.append(epoch_loss)
        val_output_loss_list.append(epoch_output_loss)
        val_dec_output_loss_list.append(epoch_dec_output_loss)
        val_stop_token_loss_list.append(epoch_stop_token_loss)

        scheduler.step()

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss_list=train_loss_list,
            train_output_loss_list=train_output_loss_list,
            train_dec_output_loss_list=train_dec_output_loss_list,
            train_stop_token_loss_list=train_stop_token_loss_list,
            val_loss_list=val_loss_list,
            val_output_loss_list=val_output_loss_list,
            val_dec_output_loss_list=val_dec_output_loss_list,
            val_stop_token_loss_list=val_stop_token_loss_list,
            epoch=current_epoch,
            ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
        )
        
        save_loss(train_loss_list, val_loss_list, save_path, "total_loss")
        save_loss(train_output_loss_list, val_output_loss_list, save_path, "output_loss")
        save_loss(train_dec_output_loss_list, val_dec_output_loss_list, save_path, "dec_output_loss")
        save_loss(train_stop_token_loss_list, val_stop_token_loss_list, save_path, "stop_token_loss")
        
        generate_for_tts(
            cfg = cfg,
            model = model,
            test_loader = test_loader,
            dataset=train_dataset,
            device=device,
            save_path=save_path,
            epoch=epoch
        )
        torch.cuda.empty_cache()
        gc.collect()

    # モデルの保存
    model_save_path = save_path / f"model_{cfg.model.name}.pth"
    torch.save(model.state_dict(), str(model_save_path))
    
if __name__=='__main__':
    main()
    
    