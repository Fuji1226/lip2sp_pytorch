from omegaconf import DictConfig, OmegaConf
import hydra

# import wandb
# wandb.init(
#     project='llip2sp_pytorch',
#     name="desk-test"
# )

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_no_chainer import KablabDataset, KablabTransform
from model.models import Lip2SP
from loss import masked_mse, delta_loss, ls_loss, fm_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator
from train import make_test_loader
from data_process.feature import mel2wave


def generate(model, test_loader, device):
    outputs = []
    dec_outputs = []

    for batch in test_loader:
        model.eval()

        (lip, target, feat_add), data_len = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)
        
        with torch.no_grad():
            output, dec_output = model.inference(
                lip=lip
            )
        outputs.append(output)
        dec_outputs.append(dec_output)
    return outputs, dec_outputs


def data_check(test_loader, cfg):
    save_path = "/users/minami/dataset"
    idx = 0
    for batch in test_loader:
        (lip, target, feat_add), data_len = batch
        lip = lip.to('cpu').detach().numpy().copy()
        target = target.to('cpu').detach().numpy().copy()
        lip = lip.squeeze(0)[:3, ...]
        target = target.squeeze(0)
        print(lip.shape)
        print(target.shape)
        wav = mel2wave(target, cfg.model.sampling_rate, cfg.model.frame_period)
        print(wav.shape)
        write(save_path+f"/out{idx}.wav", rate=cfg.model.sampling_rate, data=wav)
        idx += 1


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    ###ここにデータセットモデルのインスタンス作成train関数を回す#####

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    #インスタンス作成
    model = Lip2SP(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        glu_inner_channels=cfg.model.glu_inner_channels,
        glu_layers=cfg.model.glu_layers,
        pre_in_channels=cfg.model.pre_in_channels,
        pre_inner_channels=cfg.model.pre_inner_channels,
        post_inner_channels=cfg.model.post_inner_channels,
        n_position=cfg.model.length * 5,
        max_len=cfg.model.length // 2,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        training_method=cfg.train.training_method,
        num_passes=cfg.train.num_passes,
        mixing_prob=cfg.train.mixing_prob,
        dropout=cfg.train.dropout,
        reduction_factor=cfg.model.reduction_factor,
        use_gc=cfg.train.use_gc
    )
    model = model.to(device)

    # 保存したパラメータの読み込み
    model.load_state_dict(torch.load(cfg.model.save_path+'/model.pth'))

    # Dataloader作成
    test_loader = make_test_loader(cfg)

    # generate
    output, dec_output = generate(
        model=model,
        test_loader=test_loader,
        device=device,
    )
    


    


if __name__ == "__main__":
    main()