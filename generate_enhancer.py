"""
保存したパラメータを読み込み,合成を行う際に使用してください
まだ色々確認の段階で,printとかがたくさんあってとても汚くなってます…
一旦モデルのパラメータをロードし,音声合成を行うことは可能です

1. model_pathの変更
    main()の中にあります
    自分が読み込みたいやつまでのパスを通してください(checkpointやresult/train)

2. conf/testの変更
    顔と口唇のどちらのデータを使うか
    学習用データとテスト用データのどちらを使うか
    など

    顔か口唇かを学習時の状態に合わせれば基本的には大丈夫だと思います

3. generate.pyの実行

4. 結果の確認
    result/generateの下に保存されると思います
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
from train_default import make_model
from train_enhancer import make_enhancer


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
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=1,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, dataset


def generate(cfg, model, enhancer, test_loader, datasets, device, save_path):
    index = 0

    lip_mean = datasets.lip_mean.to(device)
    lip_std = datasets.lip_std.to(device)
    feat_mean = datasets.feat_mean.to(device)
    feat_std = datasets.feat_std.to(device)
    feat_add_mean = datasets.feat_add_mean.to(device)
    feat_add_std = datasets.feat_add_std.to(device)

    for batch in test_loader:
        model.eval()
        index += 1

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        with torch.no_grad():
            output, dec_output, enc_output = model(lip)
            enhanced_output = enhancer(output)

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
            enhanced_output=enhanced_output,
        )
        plt.close()
        if index > 4:
            break


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # 口唇動画か顔かの選択
    lip_or_face = cfg.test.face_or_lip
    assert lip_or_face == "face" or "lip"
    if cfg.test.which_data == "test":
        if lip_or_face == "face":
            data_path = cfg.test.face_pre_loaded_path
            mean_std_path = cfg.test.face_mean_std_path
        elif lip_or_face == "lip":
            data_path = cfg.test.lip_pre_loaded_path
            mean_std_path = cfg.test.lip_mean_std_path

    elif cfg.test.which_data == "train":
        if lip_or_face == "face":
            data_path = cfg.train.face_pre_loaded_path
            mean_std_path = cfg.train.face_mean_std_path
        elif lip_or_face == "lip":
            data_path = cfg.train.lip_pre_loaded_path
            mean_std_path = cfg.train.lip_mean_std_path

    print("--- data directory check ---")
    print(f"data_path = {data_path}")
    print(f"mean_std_path = {mean_std_path}")

    model = make_model(cfg, device)
    enhancer = make_enhancer(cfg, device)

    # パラメータのロード
    # model_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/face/2022:07:08_20-10-06/mspec_30.ckpt"     # glu
    # model_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/face/2022:07:07_00-58-32/mspec_80.ckpt"     # transformer
    # model.load_state_dict(torch.load(model_path)['model'])

    model_path = "/home/usr4/r70264c/lip2sp_pytorch/result/default/train/lip/2022:07:11_07-42-22/model_mspec.pth"     # glu
    model.load_state_dict(torch.load(model_path))

    enhancer_path = "/home/usr4/r70264c/lip2sp_pytorch/result/enhancer/train/lip/2022:07:13_08-13-30/enhancer_mspec.pth"
    enhancer.load_state_dict(torch.load(enhancer_path))

    # 保存先
    save_path = cfg.test.generate_save_path
    save_path = os.path.join(save_path, lip_or_face, Path(model_path).parents[0].name, Path(model_path).stem, cfg.test.which_data, cfg.test.generate_method)
    os.makedirs(save_path, exist_ok=True)

    # Dataloader作成
    test_loader, datasets = make_test_loader(cfg, data_path, mean_std_path)

    # generate
    model.eval()
    generate(
        cfg=cfg,
        model=model,
        enhancer=enhancer,
        test_loader=test_loader,
        datasets=datasets,
        device=device,
        save_path=save_path,
    )
    

if __name__ == "__main__":
    main()
    # test()