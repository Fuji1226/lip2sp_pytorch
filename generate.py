"""
保存したパラメータを読み込み,合成を行う際に使用してください

1. model_pathの変更
    main()の中にあります
    自分が読み込みたいやつまでのパスを通してください(checkpointやresult/train)

2. conf/testの変更
    顔か口唇かを学習時の状態に合わせれば基本的には大丈夫だと思います

3. generate.pyの実行

4. 結果の確認
    result/generateの下に保存されると思います
"""

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
from torch.utils.data import DataLoader

from dataset.dataset_npz import KablabDataset, KablabTransform, get_datasets
from data_check import save_data
from train_default import make_model
from calc_accuracy import calc_accuracy

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def make_test_loader(cfg, data_root, mean_std_path):
    test_data_path = get_datasets(
        data_root=data_root,
        name=cfg.model.name,
    )
    test_trans = KablabTransform(
        cfg=cfg,
        train_val_test="test",
    )
    test_dataset = KablabDataset(
        data_path=test_data_path,
        mean_std_path = mean_std_path,
        transform=test_trans,
        cfg=cfg,
        test=False,     # Falseで
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, test_dataset


def get_path(cfg, model_path):
    if cfg.test.face_or_lip == "face":
        train_data_root = cfg.train.face_pre_loaded_path
        test_data_root = cfg.test.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    if cfg.test.face_or_lip == "lip":
        train_data_root = cfg.train.lip_pre_loaded_path
        test_data_root = cfg.test.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    if cfg.test.face_or_lip == "lip_128128":
        train_data_root = cfg.train.lip_pre_loaded_path_128128
        test_data_root = cfg.test.lip_pre_loaded_path_128128
        mean_std_path = cfg.train.lip_mean_std_path_128128
    if cfg.test.face_or_lip == "lip_9696":
        train_data_root = cfg.train.lip_pre_loaded_path_9696
        test_data_root = cfg.test.lip_pre_loaded_path_9696
        mean_std_path = cfg.train.lip_mean_std_path_9696
    if cfg.test.face_or_lip == "lip_9696_time_only":
        train_data_root = cfg.train.lip_pre_loaded_path_9696_time_only
        test_data_root = cfg.test.lip_pre_loaded_path_9696_time_only
        gen_data_root = cfg.test.gen_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path_9696_time_only
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()
    gen_data_root = Path(gen_data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()
    data_root_list = [train_data_root, test_data_root, gen_data_root]

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem
    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    gen_save_path = save_path / "gen_data" / "audio"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(gen_save_path, exist_ok=True)
    save_path_list = [train_save_path, test_save_path, gen_save_path]

    data_root_list = [train_data_root, test_data_root, gen_data_root]
    save_path_list = [train_save_path, test_save_path, gen_save_path]
    print(data_root_list, save_path_list)

    return data_root_list, mean_std_path, save_path_list


def generate(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        start_time = time.time()

        with torch.no_grad():
            output, dec_output, feat_add_out = model(lip)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / label[0]
        os.makedirs(_save_path, exist_ok=True)
        
        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        iter_cnt += 1
        if iter_cnt == 53:
            break

    return process_times

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)

    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/default/lip_9696_time_only/2022:08:21_14-06-39/mspec80_300.ckpt")

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

    data_root_list, mean_std_path, save_path_list = get_path(cfg, model_path)

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

        print("--- generate ---")
        process_times = generate(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )
        
        print("--- calc accuracy ---")
        calc_accuracy(save_path, save_path.parents[0], cfg, process_times)

if __name__ == "__main__":
    main()