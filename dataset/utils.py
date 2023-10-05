from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import joblib
from functools import partial
import torch
import pickle

text_dir = Path("~/dataset/lip/utt").expanduser()
emb_dir = Path("~/dataset/lip/emb").expanduser()


def get_speaker_idx(data_path):
    print("\nget speaker idx")
    speaker_idx = {}
    idx_dict = {
        "F01_kablab" : 0,
        "F02_kablab" : 1,
        "M01_kablab" : 2,
        "M04_kablab" : 3,
        "F01_kablab_all" : 4,
    }
    for path in data_path:
        speaker = path.parents[1].name
        if speaker in speaker_idx:
            continue
        else:
            if speaker in idx_dict:
                speaker_idx[speaker] = idx_dict[speaker]
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def get_stat_load_data(train_data_path):
    print("\nget stat")
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))
        lip = npz_key['lip']
        feature = npz_key['feature']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])
        
    return (
        lip_mean_list,
        lip_var_list,
        lip_len_list,
        feat_mean_list,
        feat_var_list,
        feat_len_list,
    )
                

def calc_mean_var_std(mean_list, var_list, len_list):
    mean_square_list = list(np.square(mean_list))

    square_mean_list = []
    for var, mean_square in zip(var_list, mean_square_list):
        square_mean_list.append(var + mean_square)

    mean_len_list = []
    square_mean_len_list = []
    for mean, square_mean, len in zip(mean_list, square_mean_list, len_list):
        mean_len_list.append(mean * len)
        square_mean_len_list.append(square_mean * len)

    mean = sum(mean_len_list) / sum(len_list)
    var = sum(square_mean_len_list) / sum(len_list) - mean ** 2
    std = np.sqrt(var)
    return mean, var, std


def get_utt(data_path):
    print("--- get utterance ---")
    path_text_list = []
    for path in tqdm(data_path):
        text_path = text_dir / f"{path.stem}.txt"
        df = pd.read_csv(str(text_path), header=None)
        text = df[0].values[0]
        path_text_list.append([path, text])
    return path_text_list


def get_spk_emb(cfg):
    spk_emb_dict = {}
    for speaker in cfg.train.speaker:
        data_path = emb_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict


def adjust_max_data_len(data):
    """
    minibatchの中で最大のdata_lenに合わせて0パディングする
    data : (..., T)
    """
    max_data_len = 0
    max_data_len_id = 0

    # minibatchの中でのdata_lenの最大値と，そのデータのインデックスを取得
    for idx, d in enumerate(data):
        if max_data_len < d.shape[-1]:
            max_data_len = d.shape[-1]
            max_data_len_id = idx

    new_data = []

    # data_lenが最大のデータに合わせて0パディング
    for d in data:
        d_padded = torch.zeros_like(data[max_data_len_id])

        for t in range(d.shape[-1]):
            d_padded[..., t] = d[..., t]

        new_data.append(d_padded)
    
    return new_data