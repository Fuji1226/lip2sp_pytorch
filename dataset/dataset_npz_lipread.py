import os
import sys

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

import random
import pyopenjtalk

from dataset.phoneme_encode import *
from dataset.tts_utils import *


def get_datasets(data_root, cfg):    
    """
    npzファイルのパス取得
    """
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.train.speaker:
        print(f"load {speaker}")
        spk_path = data_root / speaker
        spk_path = list(spk_path.glob(f"*{cfg.model.name}.npz"))
        items += spk_path
    return items

class KablabLipReadDataset(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg
        self.class_to_id, self.id_to_class = classes2index_tts()


        # 統計量から平均と標準偏差を求める
        lip_mean_list, lip_var_list, lip_len_list, \
            feat_mean_list, feat_var_list, feat_len_list, \
                feat_add_mean_list, feat_add_var_list, feat_add_len_list, \
                    landmark_mean_list, landmark_var_list, landmark_len_list = get_stat_load_data(train_data_path)
        
        lip_mean, _, lip_std = calc_mean_var_std(lip_mean_list, lip_var_list, lip_len_list)
        feat_mean, _, feat_std = calc_mean_var_std(feat_mean_list, feat_var_list, feat_len_list)

        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)
        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
     
        self.path_text_label_list = get_utt_label(data_path, cfg)

        # self.lip_mean = torch.from_numpy(lip_mean)
        # self.lip_std = torch.from_numpy(lip_std)
        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path, text, label = self.path_text_label_list[index]
        
        speaker = data_path.parents[1].name
        label = torch.tensor([label]).to(torch.float)
        filename = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])
        lip_len = torch.Tensor(lip.shape[-1])
        
        feature, text, lip = self.transform(
            feature=feature,
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std,
            lip=lip,
            lip_mean = self.lip_mean,
            lip_std = self.lip_std,
            text=text,
            class_to_id=self.class_to_id,
        )

        feature_len = torch.tensor(feature.shape[-1])
        text_len = torch.tensor(text.shape[0])
        stop_token = torch.zeros(feature_len)
        stop_token[-2:] = 1.0
        
        output = {}
        output['lip'] = lip
        output['feature'] = feature
        output['feat_add'] = feat_add
        output['upsample'] = upsample
        output['data_len'] = data_len
        output['label'] = filename
        output['stop_token'] = stop_token
        output['text'] = text
        output['text_len'] = text_len
        output['lip_len'] = lip_len
        
        return output
    

class KablabLipReadTransform:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test

    def feat_normalization(
        self, feature, feat_mean, feat_std):
        """
        標準化
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        landmark : (T, 2, 68)
        
        lip_mean, lip_std : (C, H, W) or (C,)
        feat_mean, feat_std, feat_add_mean, feat_add_std : (C,)
        landmark_mean, landmark_std : (2,)
        """
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
   
        feature = (feature - feat_mean) / feat_std
        return feature
    
    def lip_norm(self, lip, lip_mean, lip_std):
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        lip = (lip -lip_mean) / lip_std       
        return lip
    
    def text2index(self, text, class_to_id):
        """
        音素を数値に変換
        """
        text = pyopenjtalk.extract_fullcontext(text)
        text = pp_symbols(text)
        text = [class_to_id[t] for t in text]
        
        # text = pyopenjtalk.g2p(text)
        # text = text.split(" ")
        # text.insert(0, "sos")
        # text.append("eos")
        # text_index = [class_to_id[t] if t in class_to_id.keys() else None for t in text]
        # print(f'text idx {text_index}')
        # assert (None in text_index) is False
        return torch.tensor(text)

    def __call__(
        self, feature, feat_mean, feat_std, text, class_to_id, lip, lip_mean, lip_std):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        landmark : (T, 2, 68)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        
        # 標準化
        feature = self.feat_normalization(feature, feat_mean, feat_std)
        lip = self.lip_norm(lip, lip_mean, lip_std)

        feature = feature.to(torch.float32)
        lip = lip.to(torch.float32)
        
        text = self.text2index(text, class_to_id)
        return feature, text, lip

def collate_time_adjust_lipread(batch, cfg):

    lip = [sample['lip'] for sample in batch]
    feature = [sample['feature'] for sample in batch]
    upsample = [sample['upsample'] for sample in batch]
    data_len = [sample['data_len'] for sample in batch]
    stop_token = [sample['stop_token'] for sample in batch]
    text = [sample['text'] for sample in batch]
    text_len = [sample['text_len'] for sample in batch]
    lip_len = [sample['lip_len'] for sample in batch]
    
    lip = adjust_max_data_len(lip)
    feature = adjust_max_data_len(feature)
    text = adjust_max_data_len(text)
    stop_token = adjust_max_data_len(stop_token)
    
    
    lip = torch.stack(lip)
    feature = torch.stack(feature)
    upsample = torch.stack(upsample)
    data_len = torch.stack(data_len)
    stop_token = torch.stack(stop_token)
    text = torch.stack(text)
    text_len = torch.stack(text_len)
    lip_len = torch.stack(lip_len)
    
    output = {}
    output['lip'] = lip
    output['feature'] = feature
    output['upsample'] = upsample
    output['data_len'] = data_len
    output['stop_token'] = stop_token
    output['text'] = text
    output['text_len'] = text_len
    output['lip_len'] = lip_len
    
    return output
