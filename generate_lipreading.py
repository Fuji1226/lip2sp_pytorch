from unicodedata import name
from omegaconf import DictConfig, OmegaConf
import hydra

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
from utils import get_path_test, make_test_loader_lipreading, load_pretrained_model
from data_process.phoneme_encode import get_keys_from_value
from data_check import save_data_lipreading

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


do_greedy = False
do_beam_search = True
beam_size = 10


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
        wav, lip, feature, feat_add, text, data_len, text_len, speaker, label = batch

        lip = lip.to(device)
        text = text.to(device)
        data_len = data_len.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = text[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = text[:, 1:]    # (B, T)

        with torch.no_grad():
            output, ctc_output = model(lip, data_len, n_max_loop=500, search_method="greedy")

        _save_path = save_path / "greedy" / speaker[0] / label[0]
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
    with open(str(save_path.parents[0] / "accuracy_greedy.txt"), "a") as f:
        f.write(f"phoneme error rate = {per_mean:f}")


def recognize_beam_search(cfg, model, test_loader, dataset, device, save_path, beam_size):
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
        wav, lip, feature, feat_add, text, data_len, text_len, speaker, label = batch

        lip = lip.to(device)
        text = text.to(device)
        data_len = data_len.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = text[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = text[:, 1:]    # (B, T)

        with torch.no_grad():
            output_list, ctc_output = model(lip, data_len, n_max_loop=500, search_method="beam_search", beam_size=beam_size)

        _save_path = save_path / f"beam_search_{beam_size}" / speaker[0] / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        min_per = 100
        for output in output_list:
            phoneme_error_rate = save_data_lipreading(
                cfg=cfg,
                save_path=_save_path,
                target=phoneme_index_output[0],
                output=output,
                classes_index=classes_index,
            )
            if min_per > phoneme_error_rate:
                min_per = phoneme_error_rate
        per_list.append(min_per)

    per_mean = sum(per_list) / len(per_list)
    with open(str(save_path.parents[0] / f"accuracy_beam_search.txt"), "a") as f:
        f.write("")
        f.write(f"beam size = {beam_size}\n")
        f.write(f"phoneme error rate = {per_mean:f}")
    

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)
    
    start_epoch = 200
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        # teacher forcingだけが一番良い
        model_path = Path(f"~/lip2sp_pytorch/check_point/lipreading/face_aligned_0_50_gray/2023:01:05_16-11-35/mspec80_{num_gen_epoch}.ckpt").expanduser()  # F01
        # model_path = Path(f"~/lip2sp_pytorch/check_point/lipreading/face_aligned_0_50_gray/2023:01:09_00-30-21/mspec80_{num_gen_epoch}.ckpt").expanduser()  # F01 time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/lipreading/face_aligned_0_50_gray/2023:01:12_15-29-05/mspec80_{num_gen_epoch}.ckpt").expanduser()  # F01 ctc tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/lipreading/face_aligned_0_50_gray/2023:01:12_17-39-40/mspec80_{num_gen_epoch}.ckpt").expanduser()  # F01 ctc ss

        # multi speaker

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader_lipreading(cfg, data_root, train_data_root)
            
            if do_greedy:
                print("--- greedy search ---")
                recognize_greedy(
                    cfg=cfg,
                    model=model,
                    test_loader=test_loader,
                    dataset=test_dataset,
                    device=device,
                    save_path=save_path,
                )
            if do_beam_search:
                print("--- beam search ---")
                recognize_beam_search(
                    cfg=cfg,
                    model=model,
                    test_loader=test_loader,
                    dataset=test_dataset,
                    device=device,
                    save_path=save_path,
                    beam_size=beam_size,
                )


if __name__ == "__main__":
    main()