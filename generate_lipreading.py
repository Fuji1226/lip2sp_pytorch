from unicodedata import name
from omegaconf import DictConfig, OmegaConf
import hydra

from pathlib import Path
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from train_lipreading import make_model
from utils import get_path_test, make_test_loader_lipreading
from data_process.phoneme_encode import get_keys_from_value
from data_check import save_data_lipreading

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def recognize_greedy(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)
    classes_index = dataset.classes_index

    per_list = []

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = batch

        lip = lip.to(device)
        phoneme_index = phoneme_index.to(device)
        data_len = data_len.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = phoneme_index[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = phoneme_index[:, 1:]    # (B, T)

        with torch.no_grad():
            output = model(lip, n_max_loop=500)

        _save_path = save_path / speaker[0] / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        phoneme_error_rate = save_data_lipreading(
            cfg=cfg,
            save_path=_save_path,
            target=phoneme_index_output[0],
            output=output[0],
            classes_index=classes_index,
        )
        per_list.append(phoneme_error_rate)

    per_mean = sum(per_list) / len(per_list)
    with open(str(save_path.parents[0] / "accuracy.txt"), "a") as f:
        f.write(f"phoneme error rate = {per_mean:f}")


def recognize_beam_search(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)
    classes_index = dataset.classes_index

    per_list = []

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = batch

        lip = lip.to(device)
        phoneme_index = phoneme_index.to(device)
        data_len = data_len.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = phoneme_index[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = phoneme_index[:, 1:]    # (B, T)

        with torch.no_grad():
            output_list = model(lip, n_max_loop=500)

        _save_path = save_path / speaker[0] / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        for output in output_list:
            phoneme_error_rate = save_data_lipreading(
                cfg=cfg,
                save_path=_save_path,
                target=phoneme_index_output[0],
                output=output[0],
                classes_index=classes_index,
            )
            per_list.append(phoneme_error_rate)

    # per_mean = sum(per_list) / len(per_list)
    # with open(str(save_path.parents[0] / "accuracy.txt"), "a") as f:
    #     f.write(f"phoneme error rate = {per_mean:f}")
    


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)
    
    start_epoch = 200
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]
    for num_gen_epoch in num_gen_epoch_list:
        model_path = Path(f"~/lip2sp_pytorch/check_point/lipreading/face_aligned_0_50_gray/2023:01:01_10-31-32/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss pitch
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

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

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader_lipreading(cfg, data_root, train_data_root)

            # recognize_greedy(
            #     cfg=cfg,
            #     model=model,
            #     test_loader=test_loader,
            #     dataset=test_dataset,
            #     device=device,
            #     save_path=save_path,
            # )
            recognize_beam_search(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )


if __name__ == "__main__":
    main()