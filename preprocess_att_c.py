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

from utils import get_path_tts_train, make_train_val_loader_tts_final, make_pad_mask_tts, check_mel_default, check_attention_weight, save_loss, make_test_loader_tts
from synthesis_tts import generate_for_tts_vq

from utils import make_all_loader_tts_final

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
    
    train_loader, train_dataset = make_all_loader_tts_final(cfg, data_root)
    
    #model
    model = make_model(cfg, device)
    
    tmp = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/taco/mspec80_362.ckpt'
    checkpoint = torch.load(str(tmp))['model']
    model.load_state_dict(checkpoint)
    model.eval()
    
    att_c_save_path = Path('/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/att_c')


    for batch in train_loader:
        wav, feature, text, stop_token, feature_len, text_len, filename = batch
        
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)
        
        with torch.no_grad():
            dec_output, output, logit, att_w, att_c = model(text, text_len, feature_target=feature)
            
        tmp_path = att_c_save_path / filename[0]
        
        att_c = att_c[0].to('cpu')
        
        np.savez(
            tmp_path,
            att_c=att_c
        )
        print(f'save: {filename[0]}')
        torch.cuda.empty_cache()
        gc.collect()

    
if __name__=='__main__':
    main()
    
    