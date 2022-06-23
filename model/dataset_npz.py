"""
事前に作っておいたnpzファイルを読み込んでデータセットを作るパターンです
以前のdataset_no_chainerが色々混ざってごちゃごちゃしてきたので

data_process/make_npz.pyで事前にnpzファイルを作成

その後にデータセットを作成するという手順に分けました
"""

import os
import sys
import glob

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from tqdm import tqdm

# 自作
from get_dir import get_datasetroot, get_data_directory
from hparams import create_hparams
from transform_no_chainer import load_data
from data_process.feature import world2wav
from data_check import data_check_trans

from omegaconf import DictConfig, OmegaConf
import hydra


def get_datasets(data_root, name, debug, debug_data_len):    
    """
    npzファイルのパス取得
    """
    items = []
    if debug:
        for curdir, dir, files in os.walk(data_root):
            for file in files:
                if file.endswith(".npz"):
                    # mspecかworldかの分岐
                    if f"{name}" in Path(file).stem:
                        data_path = os.path.join(curdir, file)
                        if os.path.isfile(data_path):
                            items.append(data_path)
                            # デバッグなのでデータ数を制限
                            if len(items) > debug_data_len:
                                break
    else:
        for curdir, dir, files in os.walk(data_root):
            for file in files:
                if file.endswith(".npz"):
                    # mspecかworldかの分岐
                    if f"{name}" in Path(file).stem:
                        data_path = os.path.join(curdir, file)
                        if os.path.isfile(data_path):
                            items.append(data_path)
    return items


def get_datasets_test(data_root, name, debug, debug_data_len):    
    """
    npzファイルのパス取得
    """
    items = []
    if debug:
        for curdir, dir, files in os.walk(data_root):
            for file in files:
                if file.endswith(".npz"):
                    # mspecかworldかの分岐
                    if f"{name}" in Path(file).stem:
                        data_path = os.path.join(curdir, file)
                        if os.path.isfile(data_path):
                            items.append(data_path)
                            # デバッグなのでデータ数を制限
                            if len(items) > debug_data_len:
                                break
    else:
        for curdir, dir, files in os.walk(data_root):
            for file in files:
                if file.endswith(".npz"):
                    # mspecかworldかの分岐
                    if f"{name}" in Path(file).stem:
                        data_path = os.path.join(curdir, file)
                        if os.path.isfile(data_path):
                            items.append(data_path)
    return items


def load_mean_std(mean_std_path, name):
    """
    一応複数話者の場合は全話者の平均にできるようにやってみました
    """
    each_lip_mean = []
    each_lip_std = []
    each_feat_mean = []
    each_feat_std = []
    each_feat_add_mean = []
    each_feat_add_std = []

    # 話者ごとにリスト
    for curdir, dirs, files in os.walk(mean_std_path):
        for file in files:
            if file.endswith('.npz'):
                if f"{name}" in Path(file).stem:
                    if f"train" in Path(file).stem:
                        # mean_std_pathにtrainとtestのディレクトリを作っていないので，一旦無理やりパスを通す
                        npz_key = np.load(os.path.join(curdir, file))
                        each_lip_mean.append(torch.from_numpy(npz_key['lip_mean']))
                        each_lip_std.append(torch.from_numpy(npz_key['lip_std']))
                        each_feat_mean.append(torch.from_numpy(npz_key['feat_mean']))
                        each_feat_std.append(torch.from_numpy(npz_key['feat_std']))
                        each_feat_add_mean.append(torch.from_numpy(npz_key['feat_add_mean']))
                        each_feat_add_std.append(torch.from_numpy(npz_key['feat_add_std']))
    
    # 話者人数で割って平均
    lip_mean = sum(each_lip_mean) / len(each_lip_mean)
    lip_std = sum(each_lip_std) / len(each_lip_std)
    feat_mean = sum(each_feat_mean) / len(each_feat_mean)
    feat_std = sum(each_feat_std) / len(each_feat_std)
    feat_add_mean = sum(each_feat_add_mean) / len(each_feat_add_mean)
    feat_add_std = sum(each_feat_add_std) / len(each_feat_add_std)

    return lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std


def load_mean_std_test(mean_std_path, name):
    """
    一応複数話者の場合は全話者の平均にできるようにやってみました
    """
    each_lip_mean = []
    each_lip_std = []
    each_feat_mean = []
    each_feat_std = []
    each_feat_add_mean = []
    each_feat_add_std = []

    # 話者ごとにリスト
    for curdir, dirs, files in os.walk(mean_std_path):
        for file in files:
            if file.endswith('.npz'):
                if f"{name}" in Path(file).stem:
                    if f"test" in Path(file).stem:
                        # mean_std_pathにtrainとtestのディレクトリを作っていないので，一旦無理やりパスを通す
                        npz_key = np.load(os.path.join(curdir, file))
                        each_lip_mean.append(torch.from_numpy(npz_key['lip_mean']))
                        each_lip_std.append(torch.from_numpy(npz_key['lip_std']))
                        each_feat_mean.append(torch.from_numpy(npz_key['feat_mean']))
                        each_feat_std.append(torch.from_numpy(npz_key['feat_std']))
                        each_feat_add_mean.append(torch.from_numpy(npz_key['feat_add_mean']))
                        each_feat_add_std.append(torch.from_numpy(npz_key['feat_add_std']))
    
    # 話者人数で割って平均
    lip_mean = sum(each_lip_mean) / len(each_lip_mean)
    lip_std = sum(each_lip_std) / len(each_lip_std)
    feat_mean = sum(each_feat_mean) / len(each_feat_mean)
    feat_std = sum(each_feat_std) / len(each_feat_std)
    feat_add_mean = sum(each_feat_add_mean) / len(each_feat_add_mean)
    feat_add_std = sum(each_feat_add_std) / len(each_feat_add_std)

    return lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std


class KablabDataset(Dataset):
    def __init__(self, data_root, mean_std_path, name, train, val, transforms, cfg, debug, visualize=False):
        super().__init__()
        assert data_root is not None, "データまでのパスを設定してください"
        assert mean_std_path is not None, "平均，標準偏差までのパスを設定してください"

        self.data_root = data_root
        self.mean_std_path = mean_std_path
        self.name = name
        self.transforms = transforms
        self.cfg = cfg
        self.train = train
        self.val = val
        self.debug = debug
        self.visualize = visualize

        # 口唇動画、音声データまでのパス一覧を取得
        if train:
            self.items = get_datasets(data_root, name, debug, cfg.train.debug_data_len)
            self.len = len(self.items)
            self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, name)
        else:
            self.items = get_datasets_test(data_root, name, debug, cfg.test.debug_data_len)
            self.len = len(self.items)
            self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std_test(mean_std_path, name)
        
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

        data = self.transforms(
            data=data, 
            data_len=data_len, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            feat_add_mean=self.feat_add_mean, 
            feat_add_std=self.feat_add_std, 
            train=self.train, 
            val=self.val
        )
        
        lip = data[0]   # (C, W, H, T)
        feature = data[1]   # (C, T)
        feat_add = data[2]  # (C, T)
        
        ############################################
        # 可視化（多分動きません）
        ############################################
        if self.visualize:
            data_check_trans(
                cfg=self.cfg, 
                index=index, 
                data=data,
                lip_mean=self.lip_mean,
                lip_std=self.lip_std,
                feat_mean=self.feat_mean,
                feat_std=self.feat_std,
                feat_add_mean=self.feat_add_mean,
                feat_add_std=self.feat_add_std
            )
        
        return (lip, feature, feat_add), data_len, speaker, label


class KablabTransform:
    def __init__(self, length, delta=True):
        self.lip_transforms = T.Compose([
            T.ColorJitter(brightness=[0.5, 1.5], contrast=0, saturation=1, hue=0.2),    
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
            # T.RandomPerspective(distortion_scale=0.5, p=0.5),
            # T.RandomRotation(degrees=(0, 60)),
            # T.RandomPosterize(bits=3, p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            # T.RandomAutocontrast(p=0.5),  # 動的特徴量がぐちゃぐちゃになる
            # T.RandomEqualize(p=0.5),  # 動的特徴量がぐちゃぐちゃになる
            T.RandomHorizontalFlip(p=0.5),
        ])
        self.length = length
        self.delta = delta

    def normalization(self, lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        標準化
        """
        lip = T.functional.normalize(lip.float(), lip_mean, lip_std)
        feature = (feature - feat_mean) / feat_std
        feat_add = (feat_add - feat_add_mean) / feat_add_std
        return lip, feature, feat_add

    def calc_delta(self, lip):
        """
        口唇動画の動的特徴量の計算
        """
        # scipywのgaussian_filterを使用するため、一旦numpyに戻してます
        lip = lip.to('cpu').detach().numpy().copy()
        if self.delta:
            lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]
            lip_pad = lip_pad.astype(lip.dtype)
            lip_pad = gaussian_filter(
                lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)
            lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
            lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
            lip_acc = lip_pad[..., 0:-2] + \
                lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
            lip = np.vstack((lip, lip_diff, lip_acc))
            lip = torch.from_numpy(lip)
        return lip

    def time_adjust(self, lip, feature, feat_add, data_len, upsample):
        """
        フレーム数の調整
        """
        # data_lenが短い場合、0パディング
        if data_len <= self.length:
            # length分の0初期化
            lip_padded = torch.zeros(lip.shape[0], lip.shape[1], lip.shape[2], self.length // int(upsample))
            feature_padded = torch.zeros(self.length, feature.shape[1])
            feat_add_padded = torch.zeros(self.length, feat_add.shape[1])

            # 代入
            # for i in range(data_len // int(upsample)):
            for i in range(torch.div(data_len, upsample, rounding_mode='trunc')):
                lip_padded[..., i] = lip[..., i]
            for i in range(data_len):
                feature_padded[i, ...] = feature[i, ...]
                feat_add_padded[i, ...] = feat_add[i, ...]
    
            lip = lip_padded
            feature = feature_padded
            feat_add = feat_add_padded
        
        # data_lenが長い場合、ランダムにlengthだけ取り出す
        if data_len > self.length:
            # 開始時点をランダムに決める
            idx = torch.div(torch.randint(0, int(data_len) - self.length, (1,)), upsample, rounding_mode="trunc")
            # idx = torch.randint(0, int(data_len) - length, (1,)) // upsample

            # length分の取り出し
            lip = lip[..., idx:idx + torch.div(self.length, upsample, rounding_mode="trunc")]
            # lip = lip[..., idx:idx + length // upsample]
            feature = feature[idx * upsample:idx * upsample + self.length, :]
            feat_add = feat_add[idx * upsample:idx * upsample + self.length, :]

        assert lip.shape[-1] == torch.div(self.length, upsample, rounding_mode="trunc"), "lengthの調整に失敗しました"
        assert feature.shape[0] and feat_add.shape[0] == self.length, "lengthの調整に失敗しました"
        
        return lip, feature, feat_add

    def __call__(self, data, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, train, val):
        """
        train=True,val=Trueの時、data augmentationとtime_adjustを適用します
        train=False,val=Trueの時,time_adjustのみを適用します(損失計算の並列化のため)
        """
        lip = data[0]   # (C, H, W, T)
        feature = data[1]   # (T, C)
        feat_add = data[2]  #(T, C)
        upsample = data[3]
        
        if train:
            # data augmentation(学習時のみ)
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = self.lip_transforms(lip)
        else:
            lip = lip.permute(-1, 0, 1, 2)

        # normalization
        lip, feature, feat_add = self.normalization(
            lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std
        )
        lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
    
        if train or val:
            # フレーム数の調整(学習時と検証時)
            lip, feature, feat_add = self.time_adjust(lip, feature, feat_add, data_len, upsample)

        # 口唇動画の動的特徴量の計算
        lip = self.calc_delta(lip)
        
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)
        return [lip, feature.to(torch.float32), feat_add.to(torch.float32)]

