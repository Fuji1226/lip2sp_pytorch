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

from dataset.dataset_lipreading import LipReadingDataset, LipReadingTransform, collate_time_adjust_ctc, get_data_simultaneously
from data_process.phoneme_encode import IGNORE_INDEX, get_classes_ctc, get_keys_from_value
from generate import get_path
from train_nar import make_model
from data_check import save_data
from calc_accuracy import calc_accuracy


def make_test_loader(cfg, data_root, mean_std_path):
    # classesを取得するために一旦学習用データを読み込む
    data_path = get_data_simultaneously(
        data_root=data_root,
        name=cfg.model.name,
    )
    classes = get_classes_ctc(data_path) 

    # テストデータを取得
    test_data_path = get_data_simultaneously(
        data_root=data_root,
        name=cfg.model.name,
    )

    # transform
    test_trans = LipReadingTransform(
        cfg=cfg,
        train_val_test="test",
    )

    # dataset
    test_dataset = LipReadingDataset(
        data_path=test_data_path,
        mean_std_path = mean_std_path,
        transform=test_trans,
        cfg=cfg,
        test=False,
        classes=classes,
    )

    # dataloader
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
        wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device) 

        start_time = time.time()

        with torch.no_grad():
            output, feat_add_out, phoneme = model(lip=lip)

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

    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_9696_time_only/2022:08:25_17-04-10/mspec80_240.ckpt")
    
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

    for data_root, save_path in zip(data_root_list[:-1], save_path_list[:-1]):
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