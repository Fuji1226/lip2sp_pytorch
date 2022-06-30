"""
lip2sp_pytorch/conf/modelにあるyamlファイルのパスや、モデルのパラメータの読み込み先のパスを設定してから実行してください

とりあえず適当に使っているだけなので，まだ整備できてません
150行目あたりにパスを設定するところがあるので,そこを変更してください
"""


from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow

import wandb

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_npz import KablabDataset, KablabTransform
from model.models import Lip2SP
from data_check import save_data
from train_wandb import make_model


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def make_test_loader(cfg, data_path, mean_std_path):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    dataset = KablabDataset(
        data_path=data_path,
        mean_std_path=mean_std_path,
        name=cfg.model.name,
        train=False,
        transform=trans,
        cfg=cfg,
        debug=cfg.test.debug
    )
    # 学習用データで確認するとき用
    # dataset = KablabDataset(
    #     data_root=cfg.train.pre_loaded_path,    # npzファイルまでのパス
    #     mean_std_path=cfg.train.mean_std_path,
    #     name=cfg.model.name,
    #     train=True,
    #     transform=trans,
    #     cfg=cfg,
    #     debug=cfg.test.debug
    # )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=1,   
        shuffle=True,
        num_workers=os.cpu_count(),      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, dataset


def generate(cfg, model, test_loader, datasets, device, save_path):
    outputs = []
    dec_outputs = []
    index = 0

    lip_mean = datasets.lip_mean.to(device)
    lip_std = datasets.lip_std.to(device)
    feat_mean = datasets.feat_mean.to(device)
    feat_std = datasets.feat_std.to(device)
    feat_add_mean = datasets.feat_add_mean.to(device)
    feat_add_std = datasets.feat_add_std.to(device)

    for batch in test_loader:
        model.eval()

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        with torch.no_grad():
            output, dec_output = model.inference(
                lip=lip
            )
        outputs.append(output)
        dec_outputs.append(dec_output)

        # ディレクトリ作成
        input_save_path = os.path.join(save_path, label[0], 'input')
        output_save_path = os.path.join(save_path, label[0], 'output')
        os.makedirs(input_save_path, exist_ok=True)
        os.makedirs(output_save_path, exist_ok=True)
        
        save_data(
            cfg=cfg,
            input_save_path=input_save_path,
            output_save_path=output_save_path,
            index=index,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            output=output,
            dec_output=dec_output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        if index > 3:
            break

        index += 1


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # 口唇動画か顔かの選択
    lip_or_face = cfg.test.face_or_lip
    assert lip_or_face == "face" or "lip"
    if lip_or_face == "face":
        data_path = cfg.test.face_pre_loaded_path
        mean_std_path = cfg.test.face_mean_std_path
    elif lip_or_face == "lip":
        data_path = cfg.test.lip_pre_loaded_path
        mean_std_path = cfg.test.lip_mean_std_path

    print("--- data directory check ---")
    print(f"data_path = {data_path}")
    print(f"mean_std_path = {mean_std_path}")

    #インスタンス作成
    model = make_model(cfg, device)

    # パラメータのロード
    model_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/lip/2022:06:30_15-44-39/mspec_50.ckpt"
    model.load_state_dict(torch.load(model_path)['model'])

    # model_path = "/home/usr4/r70264c/lip2sp_pytorch/result/train/2022:06:23_10-07-13/model_mspec.pth"
    # model.load_state_dict(torch.load(model_path))

    # 保存先
    save_path = cfg.test.generate_save_path
    # save_path = os.path.join(save_path, Path(cfg.test.model_save_path).name)
    save_path = os.path.join(save_path, lip_or_face, Path(model_path).parents[0].name, Path(model_path).stem)
    os.makedirs(save_path, exist_ok=True)

    # Dataloader作成
    test_loader, datasets = make_test_loader(cfg, data_path, mean_std_path)

    # generate
    model.eval()
    generate(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        datasets=datasets,
        device=device,
        save_path=save_path,
    )
    

if __name__ == "__main__":
    main()
    # test()