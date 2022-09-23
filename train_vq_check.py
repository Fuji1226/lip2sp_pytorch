from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from librosa.display import specshow

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import make_train_val_loader, save_loss, get_path_train, check_feat_add, check_mel_nar
from train_vq_audio import make_model
from loss import MaskedLoss
from model.nar_decoder import FeadAddPredicter
from model.classifier import SpeakerClassifier

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)



def train_one_epoch(feat_add_predicter, classifier, train_loader, optimizer_feat_add, optimizer_classifier, loss_f, cfg, device):
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    feat_add_predicter.train()
    classifier.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        with torch.no_grad():
            output, _, phoneme, spk_emb, quantize, embed_idx, vq_loss, enc_output, idx_pred, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature, data_len=data_len)

        speaker_pred = classifier(quantize)
        feat_add_pred = feat_add_predicter(out_upsample)

    return


def val_one_epoch():
    return


def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
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
    torch.backends.cudnn.deterministic = True

    # path
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    loss_f = MaskedLoss()

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_lip"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        vcnet, lip_enc = make_model(cfg, device)
        model_path = Path(cfg.train.model_path).expanduser()

        if model_path.suffix == ".ckpt":
            try:
                vcnet.load_state_dict(torch.load(str(model_path))['vcnet'])
            except:
                vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['vcnet'])
        elif model_path.suffix == ".pth":
            try:
                vcnet.load_state_dict(torch.load(str(model_path)))
            except:
                vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

        feat_add_predicter = FeadAddPredicter(
            in_channels=cfg.model.tc_inner_channels, 
            out_channels=cfg.model.tc_feat_add_channels, 
            kernel_size=3, 
            n_layers=cfg.model.tc_feat_add_layers, 
            dropout=cfg.train.dec_dropout,
        ).to(device)
        classifier = SpeakerClassifier(cfg.model.vq_emb_dim, 512, n_layers=2, bidirectional=True, n_speaker=len(cfg.train.speaker)).to(device)

        optimizer_feat_add = torch.optim.Adam(
            params=feat_add_predicter.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )
        optimizer_classifier = torch.optim.Adam(
            params=classifier.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        last_epoch = 0

        wandb.watch(feat_add_predicter, **cfg.wandb_conf.watch)
        wandb.watch(classifier, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")


