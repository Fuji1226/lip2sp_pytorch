"""
最終盤
"""

import os
import sys
from functools import partial
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


def get_datasets_test(data_root, cfg):
    """
    npzファイルのパス取得
    """
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.test.speaker:
        print(f"load {speaker}")
        spk_path = data_root / speaker
        spk_path = list(spk_path.glob(f"*{cfg.model.name}.npz"))
        items += spk_path
    return items


def get_speaker_idx(data_path):
    print("\nget speaker idx")
    speaker_idx = {}
    idx = 0
    for path in sorted(data_path):
        speaker = path.parents[0].name
        if speaker in speaker_idx:
            continue
        else:
            speaker_idx[speaker] = idx
            idx += 1
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def load_mean_std(mean_std_path, cfg):
    """
    一応複数話者の場合は全話者の平均にできるようにやってみました
    """
    print("\nload mean std")
    each_lip_mean = []
    each_lip_std = []
    each_feat_mean = []
    each_feat_std = []
    each_feat_add_mean = []
    each_feat_add_std = []

    for speaker in cfg.train.speaker:
        print(f"load {speaker}")
        spk_path = mean_std_path / speaker / f"train_{cfg.model.name}.npz"
        npz_key = np.load(str(spk_path))
        each_lip_mean.append(torch.from_numpy(npz_key['lip_mean']))
        each_lip_std.append(torch.from_numpy(npz_key['lip_std']))
        each_feat_mean.append(torch.from_numpy(npz_key['feat_mean']))
        each_feat_std.append(torch.from_numpy(npz_key['feat_std']))
        each_feat_add_mean.append(torch.from_numpy(npz_key['feat_add_mean']))
        each_feat_add_std.append(torch.from_numpy(npz_key['feat_add_std']))

    lip_mean = sum(each_lip_mean) / len(each_lip_mean)
    lip_std = sum(each_lip_std) / len(each_lip_std)
    feat_mean = sum(each_feat_mean) / len(each_feat_mean)
    feat_std = sum(each_feat_std) / len(each_feat_std)
    feat_add_mean = sum(each_feat_add_mean) / len(each_feat_add_mean)
    feat_add_std = sum(each_feat_add_std) / len(each_feat_add_std)

    return lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std


class KablabDatasetMLMFinal(Dataset):
    def __init__(self, data_path, mean_std_path, cfg):
        super().__init__()
        self.data_path = data_path

        self.speaker_idx = get_speaker_idx(data_path)
        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, cfg)
        
        self.class_to_id, self.id_to_class = classes2index_tts()
        self.path_text_label_list = get_utt_label(data_path, cfg)
        
        self.av_huvert_path = Path(cfg.train.avhubert_path)
        self.vq_idx = Path(cfg.train.vq_idx)
        
        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path, text, label = self.path_text_label_list[index]
    
        speaker = data_path.parents[0].name
        
        speaker = torch.tensor(self.speaker_idx[speaker])
        label = data_path.stem
    
        # av_hubert_label = label.replace('_0_mspec80', '')
        # av_huvert_path = self.av_huvert_path / f'{av_hubert_label}.npy'
        # av_huvert = np.load(str(av_huvert_path))
        # av_huvert = torch.from_numpy(av_huvert)
        
        vq_path = self.vq_idx / f'{label}.npz'
        encoding = np.load(str(vq_path))['encoding_idx']
        encoding = torch.from_numpy(encoding)
        data_len = torch.tensor(encoding.size())
        
        output = {}
        output['encoding'] = encoding
        output['data_len'] = data_len
        return output
    
    
      

def collate_vq_encoding_final(batch, cfg):
    """
    フレーム数の調整を行う
    """
    encoding = [sample['encoding'] for sample in batch]
    data_len = [sample['data_len'] for sample in batch]

    encoding = adjust_max_data_len(encoding)
    
    encoding = torch.stack(encoding)
    data_len = torch.stack(data_len)

    output = {}
    output['encoding'] = encoding
    output['data_len'] = data_len
    return output

def adjust_max_data_len(data):
    """
    minibatchの中で最大のdata_lenに合わせて0パディングする
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
        for t in range(d.shape[-1], d_padded.shape[-1]):
            d_padded[..., t] = -1

        new_data.append(d_padded)
    
    return new_data

def adjust_max_data_len_avhubert(data):
    """
    minibatchの中で最大のdata_lenに合わせて0パディングする
    """
    max_data_len = 0
    max_data_len_id = 0

    # minibatchの中でのdata_lenの最大値と，そのデータのインデックスを取得
    for idx, d in enumerate(data):
        if max_data_len < d.shape[0]:
            max_data_len = d.shape[0]
            max_data_len_id = idx

    new_data = []

    # data_lenが最大のデータに合わせて0パディング
    for d in data:
        d_padded = torch.zeros_like(data[max_data_len_id])

        for t in range(d.shape[0]):
            d_padded[t, ...] = d[t, ...]

        new_data.append(d_padded)
    
    return new_data

def make_train_val_loader_mlm(cfg, data_root, mean_std_path):
    # パスを取得
    data_path = get_datasets_re(
        data_root=data_root,
        cfg=cfg,
    )
    
    data_path = random.sample(data_path, len(data_path))
    n_samples = len(data_path)
    train_size = int(n_samples * 0.95)
    train_data_path = data_path[:train_size]
    val_data_path = data_path[train_size:]

    # 学習用，検証用それぞれに対してtransformを作成

    # dataset作成
    print("\n--- make train dataset ---")
    train_dataset = KablabDatasetMLMFinal(
        data_path=train_data_path,
        mean_std_path = mean_std_path,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = KablabDatasetMLMFinal(
        data_path=val_data_path,
        mean_std_path=mean_std_path,
        cfg=cfg,
    )

    # それぞれのdata loaderを作成
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_vq_encoding_final, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_vq_encoding_final, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset

def get_datasets_re(data_root, cfg):    
    """
    npzファイルのパス取得
    """
    print(f'train {data_root}')
    print("\n--- get datasets ---")
    items = []
    target_extension = '.npz'
    
    for root in data_root:
        if cfg.train.corpus is not None:
            if cfg.train.corpus=='ATR':
                file_paths = [p for p in root.rglob('*') if p.is_file() and p.suffix == target_extension and 'ATR' in str(p)]
                n_ATR = len(file_paths)
                val_tmp = [p for p in root.rglob('*') if p.is_file() and p.suffix == target_extension and 'ATR' not in str(p)]
                file_paths += val_tmp[:int(n_ATR*0.2)]

            else:
                file_paths = [p for p in root.rglob('*') if p.is_file() and p.suffix == target_extension]
        else:
            file_paths = [p for p in root.rglob('*') if p.is_file() and p.suffix == target_extension]
            
      
            if cfg.train.data_size is not None:
                all_data_size = int(cfg.train.data_size * 1.2) if cfg.train.data_size * 1.2 < len(file_paths) else len(file_paths)
                file_paths = file_paths[:all_data_size]
    
        print(f'test  {root}: {len(file_paths)}')
        items += file_paths
        
    return items
