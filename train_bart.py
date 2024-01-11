from omegaconf import OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
import torch
from timm.scheduler import CosineLRScheduler

from utils import (
    count_params,
    set_config,
    save_loss,
    check_mel_nar,
    fix_random_seed,
    get_path_train_raw,
    make_train_val_loader_with_external_data_raw,
)
from model.model_nar_ssl import Lip2SpeechSSL, Lip2SpeechLightWeight
from loss import MaskedLoss

wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def save_checkpoint(
    
):
    pass


def make_model(
    
):
    pass


def train_one_epoch(
    
):
    pass


def val_one_epoch(
    
):
    pass


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
    fix_random_seed(cfg.train.random_seed)
    
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    