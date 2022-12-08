"""
事前に作っておいたnpzファイルを読み込んでデータセットを作るパターンです
以前のdataset_no_chainerが色々混ざってごちゃごちゃしてきたので

data_process/make_npz.pyで事前にnpzファイルを作成

その後にデータセットを作成するという手順に分けました
"""

import os
import sys
import re
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import pickle
import random
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset

from data_check import save_lip_video
from data_process.mulaw import mulaw_quantize, inv_mulaw_quantize


def get_speaker_idx(data_path):
    """
    話者名を数値に変換し,話者IDとする
    複数話者音声合成で必要になります
    """
    print("\nget speaker idx")
    speaker_idx = {}
    idx_set = {
        "F01_kablab" : 0,
        "F02_kablab" : 1,
        "M01_kablab" : 2,
        "M04_kablab" : 3,
        "F01_kablab_fulldata" : 100,
    }
    for path in sorted(data_path):
        speaker = path.parents[0].name
        if speaker in speaker_idx:
            continue
        else:
            speaker_idx[speaker] = idx_set[speaker]
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
    feat_add_mean_list = []
    feat_add_var_list = []
    feat_add_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))
        # print(path)

        lip = npz_key['lip']
        feature = npz_key['feature']
        feat_add = npz_key['feat_add']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])

        feat_add_mean_list.append(np.mean(feat_add, axis=0))
        feat_add_var_list.append(np.var(feat_add, axis=0))
        feat_add_len_list.append(feat_add.shape[0])
        
    return lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list, feat_add_mean_list, feat_add_var_list, feat_add_len_list


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


class KablabDatasetSSL(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg

        # 話者ID
        self.speaker_idx = get_speaker_idx(data_path)

        # 統計量から平均と標準偏差を求める
        lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list, feat_add_mean_list, feat_add_var_list, feat_add_len_list = get_stat_load_data(train_data_path)
        lip_mean, _, lip_std = calc_mean_var_std(lip_mean_list, lip_var_list, lip_len_list)
        feat_mean, _, feat_std = calc_mean_var_std(feat_mean_list, feat_var_list, feat_len_list)
        feat_add_mean, _, feat_add_std = calc_mean_var_std(feat_add_mean_list, feat_add_var_list, feat_add_len_list)

        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)
        self.feat_add_mean = torch.from_numpy(feat_add_mean)
        self.feat_add_std = torch.from_numpy(feat_add_std)

        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path = self.data_path[index]
        speaker = data_path.parents[0].name

        # 話者名を話者IDに変換
        speaker = torch.tensor(self.speaker_idx[speaker])
        label = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        lip, lip_mask, data_len = self.transform(
            wav=wav,
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
        if self.cfg.train.debug:
            save_path = Path("~/lip2sp_pytorch/check/lip_augment").expanduser()
            os.makedirs(save_path, exist_ok=True)
            save_lip_video(self.cfg, save_path, lip, self.lip_mean, self.lip_std)
            print("save")
        return lip, lip_mask, upsample, data_len, speaker, label


class KablabTransformSSL:
    def __init__(self, cfg, train_val_test=None):
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

    def random_crop(self, lip, center):
        """
        ランダムクロップ
        lip : (T, C, H, W)
        """
        _, _, H, W = lip.shape
        assert H == 56 and W == 56

        if center:
            top = left = 3
        else:
            top = torch.randint(0, 8, (1,))
            left = torch.randint(0, 8, (1,))
        height = width = 48
        lip = T.functional.crop(lip, top, left, height, width)
        return lip

    def spatial_masking(self, lip, lip_mean):
        """
        空間領域におけるマスク
        lip : (T, C, H, W)
        """
        T, C, H, W = lip.shape
        input_type = lip.dtype
        lip = lip.to(torch.float32)
        unfold = nn.Unfold(kernel_size=H // self.cfg.train.spatial_divide_factor, stride=H // self.cfg.train.spatial_divide_factor)
        fold = nn.Fold(output_size=(H, W), kernel_size=H // self.cfg.train.spatial_divide_factor, stride=H // self.cfg.train.spatial_divide_factor)

        lip = unfold(lip)
        n_mask = torch.randint(0, self.cfg.train.n_spatial_mask, (1,))
        mask_idx = [i for i in range(lip.shape[-1])]
        mask_idx = random.sample(mask_idx, n_mask)

        if lip_mean.dim() == 1:
            lip_mean = lip_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)     # (1, C, 1, 1)
            lip_mean = lip_mean.expand(T, -1, H // self.cfg.train.spatial_divide_factor, H // self.cfg.train.spatial_divide_factor)
            lip_mean = lip_mean.reshape(T, -1)
        
        for i in mask_idx:
            lip[..., i] = lip_mean
        
        lip = fold(lip)
        return lip.to(input_type)

    def normalization(self, lip, lip_mean, lip_std):
        """
        標準化
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        
        lip_mean, lip_std : (C, H, W) or (C,)
        feat_mean, feat_std, feat_add_mean, feat_add_std : (C,)
        """
        # 時間方向にブロードキャストされるように次元を調整
        if lip_mean.dim() == 3:
            lip_mean = lip_mean.unsqueeze(-1)   # (C, H, W, 1)
            lip_std = lip_std.unsqueeze(-1)     # (C, H, W, 1)
        elif lip_mean.dim() == 1:
            lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
            lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)

        lip = (lip -lip_mean) / lip_std        
        return lip

    def calc_delta(self, lip):
        """
        口唇動画の動的特徴量の計算
        田口さんからの継承
        差分近似から求める感じです
        lip : (C, H, W, T)
        """
        lip = lip.to('cpu').detach().numpy().copy()

        # rgb or gray
        if lip.shape[0] == 3:
            lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]     # ここのRGBの配合の比率は謎
        elif lip.shape[0] == 1:
            lip_pad = lip

        lip_pad = lip_pad.astype(lip.dtype)
        lip_pad = gaussian_filter(lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)
        lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
        lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
        lip_acc = lip_pad[..., 0:-2] + lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
        lip = np.vstack((lip, lip_diff, lip_acc))
        lip = torch.from_numpy(lip)
        return lip

    def time_augment(self, lip, feature, feat_add, upsample, data_len, interp_mode="nearest"):
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
        # また、事前にmake_npz.pyで計算した音響特徴量は対数スケールになっているので、線形補完ではなく最近傍補完を使用しています
        feature = feature.unsqueeze(0)      # (1, C, T)
        feat_add = feat_add.unsqueeze(0)    # (1, C, T)
        feature = F.interpolate(feature, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0)    # (C, T)
        feat_add = F.interpolate(feat_add, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0)  # (C, T)
        
        # データの長さが変わったので、data_lenを更新して系列長を揃える
        data_len = torch.tensor(min(int(lip.shape[-1] * upsample), feature.shape[-1])).to(torch.int)
        lip = lip[..., :data_len // upsample]
        feature = feature[..., :data_len]
        feat_add = feat_add[..., :data_len]
        assert lip.shape[-1] == feature.shape[-1] // upsample
        return lip, feature, feat_add, data_len

    def frame_masking(self, lip, lip_mean):
        """
        口唇動画のフレームをランダムにマスキング
        動的特徴量との併用に適していないので微妙
        lip : (C, H, W, T)
        """
        C, H, W, T = lip.shape

        # 削除する割合を決定
        rate = torch.randint(0, self.cfg.train.frame_masking_rate, (1,)) / 100
        mask_idx = [i for i in range(T)]
        n_mask_frame = int(T * rate)

        # インデックス全体から削除する分だけランダムサンプリング
        mask_idx = sorted(random.sample(mask_idx, n_mask_frame))

        if lip_mean.dim() == 3:
            for i in mask_idx:
                lip[..., i] = lip_mean
        elif lip_mean.dim() == 1:
            for i in mask_idx:
                lip[..., i] = lip_mean.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W)
        return lip

    def segment_masking(self, lip, lip_mean):
        """
        口唇動画の連続したフレームをある程度まとめてマスキング
        lip : (C, H, W, T)
        """
        C, H, W, T = lip.shape

        # 最初の50フレーム(1秒分)から削除するセグメントの開始フレームを選択
        mask_start_idx = torch.randint(0, 50, (1,))
        idx = [i for i in range(T)]

        # マスクする系列長を決定
        mask_length = torch.randint(0, self.cfg.train.segment_masking_length, (1,))

        # 1秒あたりdel_seg_length分だけまとめて削除するためのインデックスを選択
        while True:
            mask_seg_idx = idx[mask_start_idx:mask_start_idx + mask_length]

            if lip_mean.dim() == 3:
                for i in mask_seg_idx:
                    lip[..., i] = lip_mean
            elif lip_mean.dim() == 1:
                for i in mask_seg_idx:
                    lip[..., i] = lip_mean.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W)

            # 次の開始フレームを50フレーム先にすることで1秒ごとになる
            mask_start_idx += 50

            # 次の削除範囲が動画自体の系列長を超えてしまうならループを抜ける
            if mask_start_idx + mask_length > T:
                break
        
        return lip

    def segment_masking_segmean(self, lip):
        """
        segment maskingと同様だが,segment内の平均値で全て埋めるところが違い
        一般的に使用されているのがこっちなのでこれが無難
        lip : (C, H, W, T)
        """
        lip_mask = lip.clone()
        C, H, W, T = lip_mask.shape

        # 最初の50フレーム(1秒分)から削除するセグメントの開始フレームを選択
        mask_start_idx = torch.randint(0, 50, (1,))
        idx = [i for i in range(T)]

        # マスクする系列長を決定
        mask_length = torch.randint(self.cfg.train.min_segment_masking_length, self.cfg.train.max_segment_masking_length, (1,))

        # 1秒あたりdel_seg_length分だけまとめて削除するためのインデックスを選択
        while True:
            mask_seg_idx = idx[mask_start_idx:mask_start_idx + mask_length]
            seg_mean = torch.mean(lip_mask[..., idx[mask_start_idx:mask_start_idx + mask_length]].to(torch.float), dim=-1).to(torch.uint8)
            for i in mask_seg_idx:
                lip_mask[..., i] = seg_mean

            # 次の開始フレームを50フレーム先にすることで1秒ごとになる
            mask_start_idx += 50

            # 次の削除範囲が動画自体の系列長を超えてしまうならループを抜ける
            if mask_start_idx + mask_length > T:
                break
        
        return lip_mask

    def __call__(self, wav, lip, feature, feat_add, upsample, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)

        # data augmentation
        if self.train_val_test == "train":
            # 見た目変換
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = self.apply_lip_trans(lip)

            if lip.shape[-1] == 56:
                if self.cfg.train.use_random_crop:
                    lip = self.random_crop(lip, center=False)
                else:
                    lip = self.random_crop(lip, center=True)

            if self.cfg.train.use_spatial_masking:
                lip = self.spatial_masking(lip, lip_mean)

            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            # 再生速度変更
            if self.cfg.train.use_time_augment:
                lip, feature, feat_add, data_len = self.time_augment(lip, feature, feat_add, upsample, data_len)

        else:
            if lip.shape[1] == 56:
                lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
                lip = self.random_crop(lip, center=True)
                lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        if self.cfg.train.debug:
            save_path = Path("~/lip2sp_pytorch/check/lip_augment_ssl").expanduser()
            os.makedirs(save_path, exist_ok=True)
            torchvision.io.write_video(
                filename=str(save_path / "lip_before.mp4"),
                video_array=lip.permute(-1, 1, 2, 0).expand(-1, -1, -1, 3).to(torch.uint8),
                fps=self.cfg.model.fps,
            )

        if self.cfg.train.which_seg_mask == "mean":
            lip_mask = self.segment_masking(lip, lip_mean)
        elif self.cfg.train.which_seg_mask == "seg_mean":
            lip_mask = self.segment_masking_segmean(lip)
        
        if self.cfg.train.debug:
            save_path = Path("~/lip2sp_pytorch/check/lip_augment_ssl").expanduser()
            os.makedirs(save_path, exist_ok=True)
            torchvision.io.write_video(
                filename=str(save_path / "lip.mp4"),
                video_array=lip.permute(-1, 1, 2, 0).expand(-1, -1, -1, 3).to(torch.uint8),
                fps=self.cfg.model.fps,
            )
            torchvision.io.write_video(
                filename=str(save_path / "lip_mask.mp4"),
                video_array=lip_mask.permute(-1, 1, 2, 0).expand(-1, -1, -1, 3).to(torch.uint8),
                fps=self.cfg.model.fps,
            )
            print("save")

        # 標準化
        lip = self.normalization(lip, lip_mean, lip_std)
        lip_mask = self.normalization(lip_mask, lip_mean, lip_std)

    
        return lip.to(torch.float32), lip_mask.to(torch.float32), data_len            


def collate_time_adjust_ssl(batch, cfg):
    """
    フレーム数の調整を行う
    """
    lip, lip_mask, upsample, data_len, speaker, label = list(zip(*batch))

    lip_adjusted = []
    lip_mask_adjusted = []

    # configで指定した範囲でフレーム数を決定
    lip_len = torch.randint(cfg.model.lip_min_frame, cfg.model.lip_max_frame, (1,)).item()
    upsample_scale = upsample[0].item()
    feature_len = int(lip_len * upsample_scale)

    for l, l_mask, d_len in zip(lip, lip_mask, data_len):
        # 揃えるlenよりも短い時は足りない分をゼロパディング
        if d_len <= feature_len:
            l_padded = torch.zeros(l.shape[0], l.shape[1], l.shape[2], lip_len)
            l_mask_padded = torch.zeros(l_mask.shape[0], l_mask.shape[1], l_mask.shape[2], lip_len)

            for t in range(l.shape[-1]):
                l_padded[..., t] = l[..., t]
                l_mask_padded[..., t] = l_mask[..., t]

            l = l_padded
            l_mask = l_mask_padded

        # 揃えるlenよりも長い時はランダムに切り取り
        else:
            lip_start_frame = torch.randint(0, l.shape[-1] - lip_len, (1,)).item()

            l = l[..., lip_start_frame:lip_start_frame + lip_len]
            l_mask = l_mask[..., lip_start_frame:lip_start_frame + lip_len]

        assert l.shape[-1] == lip_len and l_mask.shape[-1] == lip_len

        lip_adjusted.append(l)
        lip_mask_adjusted.append(l_mask)

    lip = torch.stack(lip_adjusted)
    lip_mask = torch.stack(lip_mask_adjusted)
    data_len = torch.stack(data_len)
    speaker = torch.stack(speaker)

    return lip, lip_mask, upsample, data_len, speaker, label

