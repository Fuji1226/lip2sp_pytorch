import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import pandas as pd

text_dir = Path("~/dataset/lip/utt").expanduser()
hifi_dir = Path("~/dataset/hifi/txt/parallel").expanduser()

def get_stat_load_data(train_data_path):
    print("\nget stat")
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []
    feat_add_mean_list = []
    feat_add_var_list = []
    feat_add_len_list = []
    landmark_mean_list = []
    landmark_var_list = []
    landmark_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))

        lip = npz_key['lip']
        feature = npz_key['feature']
        feat_add = npz_key['feat_add']
        # landmark = npz_key['landmark']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])

        feat_add_mean_list.append(np.mean(feat_add, axis=0))
        feat_add_var_list.append(np.var(feat_add, axis=0))
        feat_add_len_list.append(feat_add.shape[0])

        # landmark_mean_list.append(np.mean(landmark, axis=(0, 2)))
        # landmark_var_list.append(np.var(landmark, axis=(0, 2)))
        # landmark_len_list.append(landmark.shape[0])
        
    return lip_mean_list, lip_var_list, lip_len_list, \
        feat_mean_list, feat_var_list, feat_len_list, \
            feat_add_mean_list, feat_add_var_list, feat_add_len_list, \
                landmark_mean_list, landmark_var_list, landmark_len_list
                

def get_stat_load_data_hifi(train_data_path):
    print("\nget stat")
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))
        feature = npz_key['feature']
        # landmark = npz_key['landmark']

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])


        # landmark_mean_list.append(np.mean(landmark, axis=(0, 2)))
        # landmark_var_list.append(np.var(landmark, axis=(0, 2)))
        # landmark_len_list.append(landmark.shape[0])
        
    return feat_mean_list, feat_var_list, feat_len_list
     
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

def get_utt_label(data_path):
    print("--- get utterance ---")
    path_text_label_list = []

    for path in tqdm(data_path):
        text_path = text_dir / f"{path.stem}.txt"
        df = pd.read_csv(str(text_path), header=None)
        text = df[0].values[0]
        label = get_recorded_synth_label(path)
        path_text_label_list.append([path, text, label])
    return path_text_label_list

def get_utt_label_hifi(data_path):
    print("--- get utterance ---")
    path_text_label_list = []
    text_dir = hifi_dir

   
    for path in tqdm(data_path):
        text_path = text_dir / f"{path.stem}.txt"
        df = pd.read_csv(str(text_path), header=None)
        text = df[0].values[0]
        path_text_label_list.append([path, text])

    return path_text_label_list

def get_recorded_synth_label(path):
    if ("ATR" in path.stem) or ("balanced" in path.stem):
        label = 1
    elif "BASIC5000" in path.stem:
        if int(str(path.stem).split("_")[1]) > 2500:
            label = 0
        else:
            label = 1
    else:
        label = 0
    return label

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