"""
事前に作っておいたnpzファイルを読み込んでデータセットを作るパターンです
以前のdataset_no_chainerが色々混ざってごちゃごちゃしてきたので

data_process/make_npz.pyで事前にnpzファイルを作成

その後にデータセットを作成するという手順に分けました
"""

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


class KablabDataset(Dataset):
    def __init__(self, data_path, mean_std_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform

        self.speaker_idx = get_speaker_idx(data_path)
        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, cfg)
        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path = self.data_path[index]
        speaker = data_path.parents[0].name
        speaker = torch.tensor(self.speaker_idx[speaker])
        label = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        lip, feature, feat_add, data_len = self.transform(
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            upsample=upsample,
            data_len=data_len, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            feat_add_mean=self.feat_add_mean, 
            feat_add_std=self.feat_add_std, 
        )
        return wav, lip, feature, feat_add, upsample, data_len, speaker, label


class KablabTransform:
    def __init__(self, cfg, train_val_test=None):
        assert train_val_test == "train" or "val" or "test"
        self.lip_transforms = T.Compose([
            # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),     
            # T.ColorJitter(brightness=[0.5, 1.5], contrast=0, saturation=1, hue=0.2),    # 色変え
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)), # ぼかし
            # T.RandomPosterize(bits=3, p=0.5),     # 画像が気持ち悪くなる。
            # T.RandomAutocontrast(p=0.5),  # 動的特徴量がぐちゃぐちゃになる
            # T.RandomEqualize(p=0.5),  # 動的特徴量がぐちゃぐちゃになる
            T.RandomHorizontalFlip(p=0.5),  # 左右反転
            # T.RandomPerspective(distortion_scale=0.5, p=0.5),     # 視点変更。画像が汚くなる。
            # T.RandomResizedCrop(size=(48, 48), scale=(0.7, 1)),  # scaleの割合でクロップ範囲を決定し，リサイズ。精度が落ちたのでダメそう。
            # T.RandomCrop(size=(48, 48), padding=4),     # 事前にパディングした上でクロップ。これも精度が下がるので，cropするのは良くないのかも。
            T.RandomRotation(degrees=(0, 10)),     # degressの範囲で回転。これはやったほうが上がる。
        ])
        self.color_jitter = T.ColorJitter(brightness=[0.5, 1.5], contrast=0, saturation=1, hue=0.2)
        self.blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)) 
        self.horizontal_flip = T.RandomHorizontalFlip(p=0.5)
        self.rotation = T.RandomRotation(degrees=(-10, 10))
        self.pad = T.RandomCrop(size=(48, 48), padding=4)
        self.cfg = cfg
        self.train_val_test = train_val_test

    def apply_lip_trans(self, lip):
        """
        口唇動画へのデータ拡張
        lip : (T, C, H, W)

        color_jitter : 色変え
        blur : ぼかし
        horizontal_flip : 左右反転
        pad : パディングしてからクロップ
        rotation : 回転
        """
        if self.cfg.train.use_color_jitter:
            lip = self.color_jitter(lip)
        if self.cfg.train.use_blur:
            lip = self.blur(lip)
        if self.cfg.train.use_horizontal_flip:
            lip = self.horizontal_flip(lip)
        if self.cfg.train.use_pad:
            lip = self.pad(lip)
        if self.cfg.train.use_rotation:
            lip = self.rotation(lip)
        return lip

    def normalization(self, lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        標準化
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        """
        lip_mean = lip_mean.unsqueeze(-1)
        lip_std = lip_std.unsqueeze(-1)
        feat_mean = feat_mean.unsqueeze(-1)
        feat_std = feat_std.unsqueeze(-1)
        feat_add_mean = feat_add_mean.unsqueeze(-1)
        feat_add_std = feat_add_std.unsqueeze(-1)

        lip = (lip -lip_mean) / lip_std        
        feature = (feature - feat_mean) / feat_std
        feat_add = (feat_add - feat_add_mean) / feat_add_std
        return lip, feature, feat_add

    def calc_delta(self, lip):
        """
        口唇動画の動的特徴量の計算
        田口さんからの継承
        """
        # scipywのgaussian_filterを使用するため、一旦numpyに戻してます
        lip = lip.to('cpu').detach().numpy().copy()
        if self.cfg.model.delta:
            lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]
            lip_pad = lip_pad.astype(lip.dtype)
            lip_pad = gaussian_filter(lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)
            lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
            lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
            lip_acc = lip_pad[..., 0:-2] + lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
            lip = np.vstack((lip, lip_diff, lip_acc))
            lip = torch.from_numpy(lip)
        return lip

    def time_augment(self, lip, feature, feat_add, upsample, data_len):
        """
        再生速度を変更する
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        """
        # 変更する割合を決定
        rate = torch.randint(100 - self.cfg.train.time_augment_rate, 100 + self.cfg.train.time_augment_rate, (1,)) / 100
        T = feature.shape[-1]
        T_l = lip.shape[-1]

        # 口唇動画から取得するフレームを決定
        idx = torch.linspace(0, 1, int(T * rate) // upsample * upsample)
        idx_l = (idx[::upsample] * (T_l-1)).to(torch.int)
        
        # 重複したフレームを取得する、あるいはフレームを間引くことで再生速度を変更
        new_lip = []
        for i in idx_l:
            new_lip.append(lip[..., i.item()])
        lip = torch.stack(new_lip, dim=-1)

        # 音響特徴量を補完によって動画に合わせる
        # 時間周波数領域でリサンプリングを行うことで、ピッチが変わらないようにしています
        feature = feature.unsqueeze(0)      # (1, C, T)
        feat_add = feat_add.unsqueeze(0)    # (1, C, T)
        feature = F.interpolate(feature, scale_factor=rate, mode="nearest", recompute_scale_factor=False).squeeze(0)    # (C, T)
        feat_add = F.interpolate(feat_add, scale_factor=rate, mode="nearest", recompute_scale_factor=False).squeeze(0)  # (C, T)
        
        # データの長さが変わったので、data_lenを更新して系列長を揃える
        data_len = torch.tensor(min(int(lip.shape[-1] * upsample), feature.shape[-1])).to(torch.int)
        lip = lip[..., :data_len // upsample]
        feature = feature[..., :data_len]
        feat_add = feat_add[..., :data_len]
        assert lip.shape[-1] == feature.shape[-1] // upsample
        return lip, feature, feat_add, data_len

    def __call__(self, lip, feature, feat_add, upsample, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)

        # data augmentation
        if self.train_val_test == "train":
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = self.apply_lip_trans(lip)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            if self.cfg.train.use_time_augment:
                lip, feature, feat_add, data_len = self.time_augment(lip, feature, feat_add, upsample, data_len)

        # 標準化
        lip, feature, feat_add = self.normalization(
            lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std
        )

        # 口唇動画の動的特徴量の計算
        lip = self.calc_delta(lip)
    
        return lip.to(torch.float32), feature.to(torch.float32), feat_add.to(torch.float32), data_len            


def collate_time_adjust(batch, cfg):
    """
    フレーム数の調整を行う
    """
    wav, lip, feature, feat_add, upsample, data_len, speaker, label = list(zip(*batch))

    lip_adjusted = []
    feature_adjusted = []
    feat_add_adjusted = []

    # configで指定した範囲でフレーム数を決定
    lip_len = torch.randint(cfg.model.lip_min_frame, cfg.model.lip_max_frame, (1,)).item()
    upsample_scale = upsample[0].item()
    feature_len = int(lip_len * upsample_scale)

    for l, f, f_add, d_len in zip(lip, feature, feat_add, data_len):
        # 揃えるlenよりも短い時
        if d_len <= feature_len:
            l_padded = torch.zeros(l.shape[0], l.shape[1], l.shape[2], lip_len)
            f_padded = torch.zeros(f.shape[0], feature_len)
            f_add_padded = torch.zeros(f_add.shape[0], feature_len)

            for t in range(l.shape[-1]):
                l_padded[..., t] = l[..., t]
            
            for t in range(f.shape[-1]):
                f_padded[:, t] = f[:, t]
                f_add_padded[:, t] = f_add[:, t]

            l = l_padded
            f = f_padded
            f_add = f_add_padded

        # 揃えるlenよりも長い時
        else:
            lip_start_frame = torch.randint(0, l.shape[-1] - lip_len, (1,)).item()

            if random.random() < 0.25:
                lip_start_frame = 0

            feature_start_frame = int(lip_start_frame * upsample_scale)
            l = l[..., lip_start_frame:lip_start_frame + lip_len]
            f = f[:, feature_start_frame:feature_start_frame + feature_len]
            f_add = f_add[:, feature_start_frame:feature_start_frame + feature_len]

        assert l.shape[-1] == lip_len
        assert f.shape[-1] == feature_len
        assert f_add.shape[-1] == feature_len

        lip_adjusted.append(l)
        feature_adjusted.append(f)
        feat_add_adjusted.append(f_add)

    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    feat_add = torch.stack(feat_add_adjusted)
    data_len = torch.stack(data_len)
    speaker = torch.stack(speaker)

    return lip, feature, feat_add, upsample, data_len, speaker, label


def collate_time_adjust_for_test(batch, cfg):
    """
    フレーム数の調整を行う
    """
    wav, lip, feature, feat_add, upsample, data_len, speaker, label = list(zip(*batch))
    breakpoint()
    wav_list = []
    lip_adjusted = []
    feature_adjusted = []
    feat_add_adjusted = []

    # configで指定した範囲でフレーム数を決定
    lip_len = torch.randint(cfg.model.lip_min_frame, cfg.model.lip_max_frame, (1,)).item()
    upsample_scale = upsample[0].item()
    feature_len = int(lip_len * upsample_scale)

    for wa, l, f, f_add, d_len in zip(wav, lip, feature, feat_add, data_len):
        # 揃えるlenよりも短い時
        if d_len <= feature_len:
            l_padded = torch.zeros(l.shape[0], l.shape[1], l.shape[2], lip_len)
            f_padded = torch.zeros(f.shape[0], feature_len)
            f_add_padded = torch.zeros(f_add.shape[0], feature_len)

            for t in range(l.shape[-1]):
                l_padded[..., t] = l[..., t]
            
            for t in range(f.shape[-1]):
                f_padded[:, t] = f[:, t]
                f_add_padded[:, t] = f_add[:, t]

            l = l_padded
            f = f_padded
            f_add = f_add_padded

        # 揃えるlenよりも長い時
        else:
            lip_start_frame = torch.randint(0, l.shape[-1] - lip_len, (1,)).item()

            if random.random() < 0.3:
                lip_start_frame = 0

            feature_start_frame = int(lip_start_frame * upsample_scale)
            l = l[..., lip_start_frame:lip_start_frame + lip_len]
            f = f[:, feature_start_frame:feature_start_frame + feature_len]
            f_add = f_add[:, feature_start_frame:feature_start_frame + feature_len]

        assert l.shape[-1] == lip_len
        assert f.shape[-1] == feature_len
        assert f_add.shape[-1] == feature_len

        lip_adjusted.append(l)
        feature_adjusted.append(f)
        feat_add_adjusted.append(f_add)
        wav_list.append(wa)

    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    feat_add = torch.stack(feat_add_adjusted)
    data_len = torch.stack(data_len)
    speaker = torch.stack(speaker)
    wav = torch.stack(wav_list)

    return wav, lip, feature, feat_add, upsample, data_len, speaker, label