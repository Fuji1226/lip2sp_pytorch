import os
import sys
import glob
import random

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from dataset_npz import KablabTransform, get_datasets, load_mean_std


class BertDataset(Dataset):
    def __init__(self, data_path, mean_std_path, transform, name, debug, cfg):
        super().__init__()
        self.items = get_datasets(data_path, name, debug, cfg.train.debug_data_len)
        self.len = len(self.items)
        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, name)
        self.transform = transform

        random.shuffle(self.items)
        print(f'Size of {type(self).__name__}: {self.len}')

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        data_path = self.items[index]
        speaker = Path(data_path).parents[0].name
        label = Path(data_path).stem

        npz_key = np.load(data_path)
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        data = [lip, feature, feat_add, upsample]

        data = self.transform(
            data=data, 
            data_len=data_len, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            feat_add_mean=self.feat_add_mean, 
            feat_add_std=self.feat_add_std, 
        )

        lip = data[0]   # (C, W, H, T)
        feature = data[1]   # (C, T)
        feat_add = data[2]  # (C, T)
        lip_cutted = data[3]    # (C, W, H, T)
        return lip, feature, feat_add, upsample, data_len, speaker, label, lip_cutted


class MySubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.items = dataset.items
        print(self.__len__())

    def __getitem__(self, idx):
        data_path = self.items[self.indices[idx]]
        return self.dataset.__getitem__(data_path, self.transform)

    def __len__(self):
        return len(self.indices)


class BertTransform(KablabTransform):
    def __init__(self, length, delta=True, train=True):
        super().__init__(length, delta)
        self.train = train
        self.lip_transforms_bert = T.Compose([
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
            T.RandomRotation(degrees=(0, 10)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def __call__(self, data, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        lip = data[0]   # (C, H, W, T)
        feature = data[1]   # (T, C)
        feat_add = data[2]  #(T, C)
        upsample = data[3]

        C, H, W, T = lip.shape

        # RGBの情報を欠落させる
        index = torch.randint(0, 3, (1,))
        use_channel = lip[index, ...]
        lip[:, ...] = use_channel
        
        if self.train:
            # data augmentation
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = self.lip_transforms(lip)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
            lip, feature, feat_add, upsample, data_len = self.time_augment(lip, feature, feat_add, upsample, data_len)
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
        else:
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            
        # 標準化
        lip, feature, feat_add = self.normalization(
            lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std
        )
        lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
    
        # フレーム数の調整
        lip, feature, feat_add = self.time_adjust(lip, feature, feat_add, data_len, upsample)

        # 動画の切り取り
        span = 30
        keep_length = 30
        index = torch.randint(keep_length, T - span - keep_length, (1,))
        lip_cutted = lip[..., index:index + span]
        lip[..., index:index + span] = 0

        # 口唇動画の動的特徴量の計算
        lip = self.calc_delta(lip)
        
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)
        return [lip, feature.to(torch.float32), feat_add.to(torch.float32), lip_cutted]