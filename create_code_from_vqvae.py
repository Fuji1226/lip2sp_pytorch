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

from utils import make_train_val_loader_final, make_test_loader_final

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
    model = VQVAE_Content_ResTC(
    )
    
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)



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
    
    train_loader, val_loader, train_dataset, val_dataset =  make_train_val_loader_final(cfg, data_root, mean_std_path)


    test_data_root = [Path(cfg.test.face_pre_loaded_path).expanduser()]
    print(f'test root: {test_data_root}')

    test_loader, test_dataset = make_test_loader_final(cfg, test_data_root, mean_std_path)
    
    cfg.wandb_conf.setup.name = cfg.tag
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    
    save_path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/vq_idx'
    save_path = Path(save_path)
    
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        vq_path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/vq_vae/code800/mspec80_80.ckpt'
    
        checkpoint = torch.load(str(vq_path))['model']
        model.load_state_dict(checkpoint)
        model.eval()
        
        embedding_weights = model.vq._embedding.weight.data.cpu().numpy()
        tmp_path = save_path / 'emb_wieghts'
        np.save(tmp_path, embedding_weights)
    
        with torch.no_grad():
            for batch in train_loader:
                feature = batch['feature'].to(device)
                data_len = batch['data_len'].to(device)
                all_output = model(feature=feature, data_len=data_len)
                
                vq = all_output['vq'][0].cpu().detach().numpy()
                label = batch['label'][0]

                encoding_idx = all_output['encoding']
                encoding_idx = encoding_idx.reshape(-1).cpu().detach().numpy()
                
                tmp_path = save_path / label
                np.savez(
                    tmp_path,
                    vq=vq,
                    encoding_idx=encoding_idx
                )
                        
            for batch in val_loader:
                feature = batch['feature'].to(device)
                data_len = batch['data_len'].to(device)
                all_output = model(feature=feature, data_len=data_len)
                
                vq = all_output['vq'][0].cpu().detach().numpy()
                label = batch['label'][0]

                encoding_idx = all_output['encoding']
                encoding_idx = encoding_idx.reshape(-1).cpu().detach().numpy()
                
                tmp_path = save_path / label
                np.savez(
                    tmp_path,
                    vq=vq,
                    encoding_idx=encoding_idx
                )
                        
            for batch in test_loader:
                feature = batch['feature'].to(device)
                data_len = batch['data_len'].to(device)
                all_output = model(feature=feature, data_len=data_len)
                
                vq = all_output['vq'][0].cpu().detach().numpy()
                label = batch['label'][0]

                encoding_idx = all_output['encoding']
                encoding_idx = encoding_idx.reshape(-1).cpu().detach().numpy()
                
                tmp_path = save_path / label
                np.savez(
                    tmp_path,
                    vq=vq,
                    encoding_idx=encoding_idx
                )


    wandb.finish()


if __name__=='__main__':
    main()