"""
chainerを使わない前処理への変更
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

# 自作
from get_dir import get_datasetroot, get_data_directory
from hparams import create_hparams
from transform_no_chainer import load_data
from data_process.feature import world2wav
from data_check import data_check_trans

from omegaconf import DictConfig, OmegaConf
import hydra

# def get_datasets(data_root, mode):
#     """
#     train用とtest用のデータのディレクトリを分けておいて、modeで分岐させる感じにしてみました
#     とりあえず
#     dataset/lip/lip_cropped         にtrain用データ
#     dataset/lip/lip_cropped_test    にtest用データを適当に置いてやってみて動きました
#     """
#     items = dict()
#     idx = 0
#     if mode == "train":
#         for curDir, Dir, Files in os.walk(data_root):
#             for filename in Files:
#                 # curDirの末尾がlip_croppedの時
#                 if curDir.endswith("lip_cropped"):
#                     # filenameの末尾（拡張子）が.mp4の時
#                     if filename.endswith(".mp4"):
#                         format = ".mp4"
#                         video_path = os.path.join(curDir, filename)
#                         # 名前を同じにしているので拡張子だけ.wavに変更
#                         audio_path = os.path.join(curDir, filename.replace(str(format), ".wav"))

#                         if os.path.isfile(audio_path) and os.path.isfile(audio_path):
#                             items[idx] = [video_path, audio_path]
#                             idx += 1
#                 else:
#                     continue
#     else:
#         for curDir, Dir, Files in os.walk(data_root):
#             for filename in Files:
#                 # curDirの末尾がlip_cropped_testの時
#                 if curDir.endswith("lip_cropped_test"):
#                     # filenameの末尾（拡張子）が.mp4の時
#                     if filename.endswith(".mp4"):
#                         format = ".mp4"
#                         video_path = os.path.join(curDir, filename)
#                         # 名前を同じにしているので拡張子だけ.wavに変更
#                         audio_path = os.path.join(curDir, filename.replace(str(format), ".wav"))

#                         if os.path.isfile(audio_path) and os.path.isfile(audio_path):
#                             items[idx] = [video_path, audio_path]
#                             idx += 1
#                 else:
#                     continue
#     return items



def get_datasets(data_root):
    """
    hparams.pyのtrain_path, test_pathからファイルを取ってくる
    """
    
    items = dict()
    idx = 0
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".mp4"):
                format = ".mp4"
                video_path = os.path.join(curdir, file)
                audio_path = os.path.join(curdir, file.replace(str(format), ".wav"))
                if os.path.isfile(audio_path) and os.path.isfile(audio_path):
                        items[idx] = [video_path, audio_path]
                        idx += 1
    return items

# def normalization(data_root=data_root, mode='train'):
#     items = get_datasets(data_root, mode)
#     data_len = len(items)
#     item_iter = iter(items)
#     item_idx = next(item_iter)

#     while item_idx:
#         video_path, audio_path = items[item_idx]


def calc_mean_var(items, len, cfg):
    lip_mean = 0
    lip_std = 0
    feat_mean = 0
    feat_std = 0
    feat_add_mean = 0
    feat_add_std = 0
    for i in range(len):
        video_path, audio_path = items[i]
        (lip, feature, feat_add, upsample), data_len = load_data(
            data_path=Path(video_path),
            gray=cfg.model.gray,
            frame_period=cfg.model.frame_period,
            feature_type=cfg.model.feature_type,
            nmels=cfg.model.n_mel_channels,
            f_min=cfg.model.f_min,
            f_max=cfg.model.f_max,
        )
        
        # 時間方向に平均と分散を計算
        lip_mean += torch.mean(lip.float(), dim=(1, 2, 3))
        lip_std += torch.mean(lip.float(), dim=(1, 2, 3))
        feat_mean += torch.mean(feature, dim=0)
        feat_std += torch.std(feature, dim=0)
        feat_add_mean += torch.mean(feat_add, dim=0)
        feat_add_std += torch.std(feat_add, dim=0)
    
    # データ全体の平均、分散を計算 (C,)
    lip_mean /= len     
    lip_std /= len      
    feat_mean /= len    
    feat_std /= len     
    feat_add_mean /= len
    feat_add_std /= len

    return lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std


class KablabDataset(Dataset):
    def __init__(self, data_root, train, transforms, cfg, visualize=False):
        super().__init__()
        assert data_root is not None, "データまでのパスを設定してください"
        self.data_root = data_root
        self.transforms = transforms
        self.cfg = cfg
        self.train = train
        self.visualize = visualize

        # 口唇動画、音声データまでのパス一覧を取得
        self.items = get_datasets(self.data_root)
        self.len = len(self.items)
        
        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = calc_mean_var(self.items, self.len, self.cfg)
        
        print(f'Size of {type(self).__name__}: {self.len}')

        # random.shuffle(self.items)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #item_idx = next(self.item_iter)
        video_path, audio_path = self.items[index]
        data_path = Path(video_path)
        label = data_path.stem

        ########################################################################################
        # 処理
        ########################################################################################
        # data = (lip, y, feat_add, upsample)
        # tensorで返してます
        data, data_len = load_data(
            data_path=Path(video_path),
            gray=self.cfg.model.gray,
            frame_period=self.cfg.model.frame_period,
            feature_type=self.cfg.model.feature_type,
            nmels=self.cfg.model.n_mel_channels,
            f_min=self.cfg.model.f_min,
            f_max=self.cfg.model.f_max,
        )

        data = self.transforms(
            data, data_len, self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std, self.train
        )
        
        lip = data[0]   # (C, W, H, T)
        feature = data[1]   # (C, T)
        feat_add = data[2]  # (C, T)
        
        ########################################################################################
        # 可視化
        ########################################################################################
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
        
        return (lip, feature, feat_add), data_len, label


class KablabTransform:
    def __init__(self, length, delta=True):
        # とりあえず適当に色々使ってみました
        self.lip_transforms = T.Compose([
            T.ColorJitter(brightness=[0.5, 1.5], contrast=0, saturation=1, hue=0.2),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
            T.RandomPerspective(distortion_scale=0.5, p=0.5),
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
        正規化
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
            lip_padded = torch.zeros(lip.shape[0], lip.shape[1], lip.shape[2], self.length // upsample)
            feature_padded = torch.zeros(self.length, feature.shape[1])
            feat_add_padded = torch.zeros(self.length, feat_add.shape[1])

            # 代入
            for i in range(data_len // upsample):
                lip_padded[..., i] = lip[..., i]
            for i in range(data_len):
                feature_padded[i, ...] = feature[..., i]
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
        
        return lip, feature.to(torch.float32), feat_add.to(torch.float32)

    def __call__(self, data, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, train):
        """
        train=Trueの時、data augmentationとtime_adjustを適用します
        """
        lip = data[0]   # (C, H, W, T)
        feature = data[1]   # (T, C)
        feat_add = data[2]  #(T, C)
        upsample = data[3]
        
        if train:
            # data augmentation
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = self.lip_transforms(lip)
        else:
            lip = lip.permute(-1, 0, 1, 2)

        # normalization
        lip, feature, feat_add = self.normalization(
            lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std
        )
        lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
    
        if train:
            # フレーム数の調整
            lip, feature, feat_add = self.time_adjust(lip, feature, feat_add, data_len, upsample)

        # 口唇動画の動的特徴量の計算
        lip = self.calc_delta(lip)
        
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)
        return [lip, feature, feat_add]



    

            

        
@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    print("############################ Start!! ############################")
    # data_root = Path(get_datasetroot()).expanduser()    

    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    datasets = KablabDataset(
        data_root=cfg.model.train_path,
        train=True,
        transforms=trans,
        cfg=cfg,
    )
    train_loader = DataLoader(
        dataset=datasets,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )


    # results
    for interation in range(1):
        for bdx, batch in enumerate(train_loader):
            (lip, y, feat_add), data_len = batch
            print("################################################")
            print(type(lip))
            print(type(y))
            print(type(feat_add))
            print(lip.dtype)
            print(y.dtype)
            print(feat_add.dtype)
            print(f"lip = {lip.shape}")  # (B, C=5, W=48, H=48, T=150)
            print(f"y(acoustic features) = {y.shape}") # (B, C, T=300)
            print(f"feat_add = {feat_add.shape}")     # (B, C=3, T=300)
            print(f"data_len = {data_len}")

    
if __name__ == "__main__":
    main()