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
from torchvision import transforms as T
from torch.utils.data import Dataset

from data_process.transform import load_data_lrs2
from dataset.utils import get_stat_load_data_lrs2, calc_mean_var_std, select_data_lrs2, get_spk_emb_lrs2
from data_process.face_crop_align import FaceAligner


class LRS2Dataset(Dataset):
    def __init__(
        self, data_root, data_bbox_root, data_landmark_root, data_df, 
        train_data_root, train_data_bbox_root, train_data_landmark_root, train_data_df, transform, cfg):
        super().__init__()
        self.transform = transform
        self.cfg = cfg
        data_path_list = select_data_lrs2(data_root, data_bbox_root, data_landmark_root, data_df, cfg)
        train_data_path_list = select_data_lrs2(train_data_root, train_data_bbox_root, train_data_landmark_root, train_data_df, cfg)
        self.data_path_list = data_path_list

        desired_left_eye = (cfg.model.align_desired_left_eye, cfg.model.align_desired_left_eye)
        desired_face_size = cfg.model.align_desired_face_size
        self.aligner = FaceAligner(desired_left_eye, desired_face_size, desired_face_size)

        lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list = \
            get_stat_load_data_lrs2(train_data_path_list, cfg, self.aligner)
        lip_mean, _, lip_std = calc_mean_var_std(lip_mean_list, lip_var_list, lip_len_list)
        feat_mean, _, feat_std = calc_mean_var_std(feat_mean_list, feat_var_list, feat_len_list)
        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)

        self.embs = get_spk_emb_lrs2()

        print(f"n = {self.__len__()}")

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        video_path, bbox_path, landmark_path = self.data_path_list[index]
        speaker = video_path.parents[0].name
        label = video_path.stem

        spk_emb = torch.from_numpy(self.embs[str(speaker)])

        wav, lip, feature, data_len = load_data_lrs2(video_path, bbox_path, landmark_path, self.cfg, self.aligner)
        wav = torch.from_numpy(wav).to(torch.float32)
        lip = torch.from_numpy(lip)
        feature = torch.from_numpy(feature)

        lip, feature = self.transform(
            lip=lip, 
            feature=feature, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std
        )
        feature_len = torch.tensor(feature.shape[-1])
        lip_len = torch.tensor(lip.shape[-1])
        return wav, lip, feature, spk_emb, feature_len, lip_len, speaker, label


class LRS2Transform:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test

    def random_crop(self, lip, center):
        """
        ランダムクロップ
        lip : (T, C, H, W)
        center : 中心を切り取るかどうか
        """
        _, _, H, W = lip.shape
        if center:
            top = left = (self.cfg.model.imsize - self.cfg.model.imsize_cropped) // 2
        else:
            top = torch.randint(0, self.cfg.model.imsize - self.cfg.model.imsize_cropped, (1,))
            left = torch.randint(0, self.cfg.model.imsize - self.cfg.model.imsize_cropped, (1,))
        height = width = self.cfg.model.imsize_cropped
        lip = T.functional.crop(lip, top, left, height, width)
        return lip

    def normalization(self, lip, feature, lip_mean, lip_std, feat_mean, feat_std):
        """
        lip : (C, H, W, T)
        feature : (C, T)
        lip_mean, lip_std, feat_mean, feat_std : (C,)
        """
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
        lip = (lip - lip_mean) / lip_std        
        feature = (feature - feat_mean) / feat_std
        return lip, feature

    def segment_masking_segmean(self, lip):
        """
        segment maskingと同様だが,segment内の平均値で全て埋めるところが違い
        一般的に使用されているのがこっちなのでこれが無難
        lip : (C, H, W, T)
        """
        C, H, W, T = lip.shape

        # 最初の1秒から削除するセグメントの開始フレームを選択
        mask_start_idx = torch.randint(0, self.cfg.model.fps, (1,))
        idx = [i for i in range(T)]

        # マスクする系列長を決定
        mask_length = torch.randint(0, int(self.cfg.model.fps * self.cfg.train.max_segment_masking_sec), (1,))

        while True:
            mask_seg_idx = idx[mask_start_idx:mask_start_idx + mask_length]
            seg_mean_lip = torch.mean(lip[..., idx[mask_start_idx:mask_start_idx + mask_length]].to(torch.float), dim=-1).to(torch.uint8)
            for i in mask_seg_idx:
                lip[..., i] = seg_mean_lip

            # 開始フレームを1秒先に更新
            mask_start_idx += self.cfg.model.fps

            # 次の範囲が動画自体の系列長を超えてしまうならループを抜ける
            if mask_start_idx + mask_length - 1 > T:
                break
        
        return lip

    def __call__(self, lip, feature, lip_mean, lip_std, feat_mean, feat_std):
        """
        lip : (C, H, W, T)
        feature : (T, C)
        """
        feature = feature.permute(1, 0)     # (C, T)
        lip = lip.permute(3, 0, 1, 2)   # (T, C, H, W)

        if self.train_val_test == "train":
            if self.cfg.train.use_random_crop:
                lip = self.random_crop(lip, center=False)
            else:
                lip = self.random_crop(lip, center=True)

            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
            if self.cfg.train.use_segment_masking:
                lip = self.segment_masking_segmean(lip)
        else:
            lip = self.random_crop(lip, center=True)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        lip, feature = self.normalization(lip, feature, lip_mean, lip_std, feat_mean, feat_std)

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        return lip, feature


def collate_time_adjust_lrs2(batch, cfg):
    wav, lip, feature, spk_emb, feature_len, lip_len, speaker, label = list(zip(*batch))

    wav_adjusted = []
    lip_adjusted = []
    feature_adjusted = []

    lip_input_len = cfg.model.input_lip_sec * cfg.model.fps
    upsample_scale = 1000 // cfg.model.frame_period // cfg.model.fps
    feat_input_len = int(lip_input_len * upsample_scale)
    wav_input_len = int(feat_input_len * cfg.model.hop_length)

    for w, l, f, f_len in zip(wav, lip, feature, feature_len):
        # 揃えるlenよりも短い時は足りない分をゼロパディング
        if f_len <= feat_input_len:
            w_padded = torch.zeros(wav_input_len)
            l_padded = torch.zeros(l.shape[0], l.shape[1], l.shape[2], lip_input_len)
            f_padded = torch.zeros(f.shape[0], feat_input_len)

            # 音響特徴量の系列長をベースに判定しているので、稀に波形のサンプル数が多い場合がある
            # その際に余ったサンプルを除外する（シフト幅的に余りが生じているのでそれを省いている）
            w = w[:wav_input_len]     

            w_padded[:w.shape[0]] = w
            l_padded[..., :l.shape[-1]] = l
            f_padded[:, :f.shape[-1]] = f

            w = w_padded
            l = l_padded
            f = f_padded

        # 揃えるlenよりも長い時はランダムに切り取り
        else:
            lip_start_frame = torch.randint(0, l.shape[-1] - lip_input_len, (1,)).item()
            feature_start_frame = int(lip_start_frame * upsample_scale)
            wav_start_sample = int(feature_start_frame * cfg.model.hop_length)

            w = w[wav_start_sample:wav_start_sample + wav_input_len]
            l = l[..., lip_start_frame:lip_start_frame + lip_input_len]
            f = f[:, feature_start_frame:feature_start_frame + feat_input_len]

        assert w.shape[0] == wav_input_len
        assert l.shape[-1] == lip_input_len
        assert f.shape[-1] == feat_input_len

        wav_adjusted.append(w)
        lip_adjusted.append(l)
        feature_adjusted.append(f)
    
    wav = torch.stack(wav_adjusted)
    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    spk_emb = torch.stack(spk_emb)
    feature_len = torch.stack(feature_len)
    lip_len = torch.stack(lip_len)
    return wav, lip, feature, spk_emb, feature_len, lip_len, speaker, label