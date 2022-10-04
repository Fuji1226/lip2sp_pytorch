import hydra

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from train_classifier import make_model
from utils import make_test_loader, get_path_test

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    acc = 0
    for batch in test_loader:
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        with torch.no_grad():
            pred = model(feature.permute(0, 2, 1))
        
        pred = F.softmax(pred, dim=-1)
        pred = torch.argmax(pred, dim=-1)

        acc += torch.sum(pred == speaker)

        print(f"pred = {pred}, ans = {speaker}, acc = {acc}/{len(test_loader)}")

    acc = acc.to(torch.float)
    acc /= float(len(test_loader))
    print(f"accuracy = {acc}")


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)

    model_path = Path("~/lip2sp_pytorch/check_point/classifier/lip/2022:10:03_00-18-07/mspec80_40.ckpt").expanduser()

    if model_path.suffix == ".ckpt":
        try:
            model.load_state_dict(torch.load(str(model_path))['model'])
        except:
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['model'])
    elif model_path.suffix == ".pth":
        try:
            model.load_state_dict(torch.load(str(model_path)))
        except:
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

    data_root_list, mean_std_path, save_path_list = get_path_test(cfg, model_path)

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

        print("--- generate ---")
        generate(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )
        

if __name__ == "__main__":
    main()