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
from torchvision import transforms as T
from torch.utils.data import Dataset

from data_check import save_lip_video, save_landmark_video
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
    landmark_mean_list = []
    landmark_var_list = []
    landmark_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))

        lip = npz_key['lip']
        feature = npz_key['feature']
        feat_add = npz_key['feat_add']
        landmark = npz_key['landmark']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])

        feat_add_mean_list.append(np.mean(feat_add, axis=0))
        feat_add_var_list.append(np.var(feat_add, axis=0))
        feat_add_len_list.append(feat_add.shape[0])

        landmark_mean_list.append(np.mean(landmark, axis=(0, 2)))
        landmark_var_list.append(np.var(landmark, axis=(0, 2)))
        landmark_len_list.append(landmark.shape[0])
        
    return lip_mean_list, lip_var_list, lip_len_list, \
        feat_mean_list, feat_var_list, feat_len_list, \
            feat_add_mean_list, feat_add_var_list, feat_add_len_list, \
                landmark_mean_list, landmark_var_list, landmark_len_list


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


class KablabDataset(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg

        # 話者ID
        self.speaker_idx = get_speaker_idx(data_path)

        # 統計量から平均と標準偏差を求める
        lip_mean_list, lip_var_list, lip_len_list, \
            feat_mean_list, feat_var_list, feat_len_list, \
                feat_add_mean_list, feat_add_var_list, feat_add_len_list, \
                    landmark_mean_list, landmark_var_list, landmark_len_list = get_stat_load_data(train_data_path)

        lip_mean, _, lip_std = calc_mean_var_std(lip_mean_list, lip_var_list, lip_len_list)
        feat_mean, _, feat_std = calc_mean_var_std(feat_mean_list, feat_var_list, feat_len_list)
        feat_add_mean, _, feat_add_std = calc_mean_var_std(feat_add_mean_list, feat_add_var_list, feat_add_len_list)
        landmark_mean, _, landmark_std = calc_mean_var_std(landmark_mean_list, landmark_var_list, landmark_len_list)

        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)
        self.feat_add_mean = torch.from_numpy(feat_add_mean)
        self.feat_add_std = torch.from_numpy(feat_add_std)
        self.landmark_mean = torch.from_numpy(landmark_mean)
        self.landmark_std = torch.from_numpy(landmark_std)

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
        landmark = torch.from_numpy(npz_key['landmark'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, data_len = self.transform(
            wav=wav,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            landmark=landmark,
            upsample=upsample,
            data_len=data_len, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            feat_add_mean=self.feat_add_mean, 
            feat_add_std=self.feat_add_std, 
            landmark_mean=self.landmark_mean,
            landmark_std=self.landmark_std,
        )
        if self.cfg.train.debug:
            save_path = Path("~/lip2sp_pytorch/check/lip_augment_now").expanduser()
            os.makedirs(save_path, exist_ok=True)
            save_lip_video(self.cfg, save_path, lip, self.lip_mean, self.lip_std)
            # save_landmark_video(landmark, save_dir=save_path)
            print("save")
        return wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label


class KablabTransform:
    def __init__(self, cfg, train_val_test):
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

    def spatial_masking(self, lip):
        """
        空間領域におけるマスク
        lip : (T, C, H, W)
        """
        T, C, H, W = lip.shape
        lip_aug = lip.clone()
        
        if self.cfg.train.which_spatial_mask == "has":
            input_type = lip.dtype
            lip_aug = lip_aug.to(torch.float32)
            unfold = nn.Unfold(kernel_size=H // self.cfg.train.spatial_divide_factor, stride=H // self.cfg.train.spatial_divide_factor)
            fold = nn.Fold(output_size=(H, W), kernel_size=H // self.cfg.train.spatial_divide_factor, stride=H // self.cfg.train.spatial_divide_factor)

            lip_aug = unfold(lip_aug)
            # n_mask = torch.randint(0, self.cfg.train.n_spatial_mask, (1,))
            # mask_idx = [i for i in range(lip_aug.shape[-1])]
            # mask_idx = random.sample(mask_idx, n_mask)

            n_mask = self.cfg.train.n_spatial_mask
            mask_idx = [i for i in range(lip_aug.shape[-1])]
            mask_idx = mask_idx[:40] + [i for i in range(56, 64, 1)] + [40, 41, 46, 47] + [48, 49, 54, 55]      # all
            # mask_idx = [i for i in range(0, 8, 1)] + [8, 9, 14, 15] + [16, 23] + [24, 31]\
            #     + [32, 33, 38, 39] +[40, 41, 46, 47] + [48, 49, 54, 55] + [i for i in range(56, 64, 1)]   # outline
            print(f"mask_index = {mask_idx}")
            
            for i in mask_idx:
                lip_aug[..., i] = 0
            
            lip_aug = fold(lip_aug).to(input_type)

        elif self.cfg.train.which_spatial_mask == "cutout":
            do_or_through = random.randint(0, 1)
            if do_or_through == 1:
                mask = torch.zeros(H, W)
                # x_center = torch.randint(0, W, (1,))
                # y_center = torch.randint(0, H, (1,))
                x_center = torch.randint(self.cfg.train.mask_length // 2, W - self.cfg.train.mask_length // 2, (1,))
                y_center = torch.randint(self.cfg.train.mask_length // 2, H - self.cfg.train.mask_length // 2, (1,))
                x1 = torch.clamp(x_center - self.cfg.train.mask_length // 2, min=0, max=W)
                x2 = torch.clamp(x_center + self.cfg.train.mask_length // 2, min=0, max=W)
                y1 = torch.clamp(y_center - self.cfg.train.mask_length // 2, min=0, max=W)
                y2 = torch.clamp(y_center + self.cfg.train.mask_length // 2, min=0, max=W)
                mask[y1:y2, x1:x2] = 1

                mask = mask.to(torch.bool)
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(lip_aug)   # (T, C, H, W)
                lip_aug = torch.where(mask, torch.zeros_like(lip_aug), lip_aug)

        return lip_aug

    def normalization(
        self, lip, feature, feat_add, landmark,
        lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, landmark_mean, landmark_std):
        """
        標準化
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        landmark : (T, 2, 68)
        
        lip_mean, lip_std : (C, H, W) or (C,)
        feat_mean, feat_std, feat_add_mean, feat_add_std : (C,)
        landmark_mean, landmark_std : (2,)
        """
        if lip_mean.dim() == 3:
            lip_mean = lip_mean.unsqueeze(-1)   # (C, H, W, 1)
            lip_std = lip_std.unsqueeze(-1)     # (C, H, W, 1)
        elif lip_mean.dim() == 1:
            lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
            lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
        feat_add_mean = feat_add_mean.unsqueeze(-1)     # (C, 1)
        feat_add_std = feat_add_std.unsqueeze(-1)       # (C, 1)
        landmark_mean = landmark_mean.unsqueeze(0).unsqueeze(-1)    # (1, 2, 1)
        landmark_std = landmark_std.unsqueeze(0).unsqueeze(-1)    # (1, 2, 1)

        lip = (lip -lip_mean) / lip_std        
        feature = (feature - feat_mean) / feat_std
        feat_add = (feat_add - feat_add_mean) / feat_add_std
        landmark = (landmark - landmark_mean) / landmark_std
        return lip, feature, feat_add, landmark

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
            # lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]     # ここのRGBの配合の比率は謎
            lip_pad = lip
        elif lip.shape[0] == 1:
            lip_pad = lip

        lip_pad = lip_pad.astype(lip.dtype)
        # lip_pad = gaussian_filter(lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)  # 平滑化？
        lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
        lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
        lip_acc = lip_pad[..., 0:-2] + lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
        lip = np.vstack((lip, lip_diff, lip_acc))
        lip = torch.from_numpy(lip)
        return lip

    def time_augment(self, lip, feature, feat_add, landmark, wav, upsample, data_len, interp_mode="nearest"):
        """
        再生速度を変更する
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        landmark : (T, 2, 68)
        wav : (T,)
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
        new_landmark = []
        for i in idx_l:
            new_lip.append(lip[..., i.item()])
            new_landmark.append(landmark[i.item(), ...])
        lip = torch.stack(new_lip, dim=-1)
        landmark = torch.stack(new_landmark, dim=0)

        # 音響特徴量を補完によって動画に合わせる
        # 時間周波数領域でリサンプリングを行うことで、ピッチが変わらないようにしています
        # また、事前にmake_npz.pyで計算した音響特徴量は対数スケールになっているので、線形補完ではなく最近傍補完を使用しています
        feature = feature.unsqueeze(0)      # (1, C, T)
        feat_add = feat_add.unsqueeze(0)    # (1, C, T)
        feature = F.interpolate(feature, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0)    # (C, T)
        feat_add = F.interpolate(feat_add, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0)  # (C, T)

        # 系列長を揃えるために音声波形にも適用しているが、音声波形はピッチが変化するので学習に使用するかどうか注意
        # メルスペクトログラムを予測する際には無関係だが、ボコーダを学習する際には使わない方がいい
        wav = wav.unsqueeze(0).unsqueeze(0)     # (1, 1, T)
        wav = F.interpolate(wav, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0).squeeze(0)   # (T,)
        
        # データの長さが変わったので、data_lenを更新して系列長を揃える
        data_len = torch.tensor(min(int(lip.shape[-1] * upsample), feature.shape[-1])).to(torch.int)
        lip = lip[..., :data_len // upsample]
        feature = feature[..., :data_len]
        feat_add = feat_add[..., :data_len]
        landmark = landmark[:data_len // upsample, ...]

        wav = wav[:int(data_len * self.cfg.model.hop_length)]
        wav_padded = torch.zeros(int(data_len * self.cfg.model.hop_length))
        wav_padded[:wav.shape[0]] = wav
        wav = wav_padded

        assert lip.shape[-1] == feature.shape[-1] // upsample
        assert lip.shape[-1] == landmark.shape[0]
        assert wav.shape[0] == int(feature.shape[-1] * self.cfg.model.hop_length)

        return lip, feature, feat_add, landmark, data_len, wav

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
            if mask_start_idx + mask_length - 1 > T:
                break
        
        return lip

    def segment_masking_segmean(self, lip, landmark):
        """
        segment maskingと同様だが,segment内の平均値で全て埋めるところが違い
        一般的に使用されているのがこっちなのでこれが無難
        lip : (C, H, W, T)
        """
        C, H, W, T = lip.shape

        # 最初の50フレーム(1秒分)から削除するセグメントの開始フレームを選択
        mask_start_idx = torch.randint(0, 50, (1,))
        idx = [i for i in range(T)]

        # マスクする系列長を決定
        mask_length = torch.randint(self.cfg.train.min_segment_masking_length, self.cfg.train.max_segment_masking_length, (1,))

        # 1秒あたりdel_seg_length分だけまとめて削除するためのインデックスを選択
        while True:
            mask_seg_idx = idx[mask_start_idx:mask_start_idx + mask_length]
            seg_mean_lip = torch.mean(lip[..., idx[mask_start_idx:mask_start_idx + mask_length]].to(torch.float), dim=-1).to(torch.uint8)
            seg_mean_landmark = torch.mean(landmark[idx[mask_start_idx:mask_start_idx + mask_length], ...].to(torch.float), dim=0).to(torch.uint8)
            for i in mask_seg_idx:
                lip[..., i] = seg_mean_lip
                landmark[i, ...] = seg_mean_landmark

            # 次の開始フレームを50フレーム先にすることで1秒ごとになる
            mask_start_idx += 50

            # 次の削除範囲が動画自体の系列長を超えてしまうならループを抜ける
            if mask_start_idx + mask_length - 1 > T:
                break
        
        return lip, landmark

    def time_frequency_masking(self, feature):
        """
        音響特徴量に対して時間周波数領域におけるマスキング
        feature : (C, T)
        """
        # 参照渡しで元の音響特徴量が変化することを防ぐため、コピーする
        feature_masked = feature.clone()

        # time masking
        time_mask_length = random.randint(0, self.cfg.train.feature_time_masking_length)
        one_second_frames = self.cfg.model.sampling_rate // self.cfg.model.hop_length
        time_mask_start_index = random.randint(0, one_second_frames - 1 - time_mask_length)
        
        while True:
            feature_masked[:, time_mask_start_index:time_mask_start_index + time_mask_length] = 0
            time_mask_start_index += one_second_frames

            if time_mask_start_index + time_mask_length - 1 > feature_masked.shape[-1]:
                break

        # frequency masking
        freq_mask_band = random.randint(0, self.cfg.train.feature_freq_masking_band)
        freq_index = random.randint(0, feature_masked.shape[0] - 1 - freq_mask_band)
        feature_masked[freq_index:self.cfg.train.feature_freq_masking_band, :] = 0

        return feature_masked

    def __call__(
        self, wav, lip, feature, feat_add, landmark, upsample, data_len,
        lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, landmark_mean, landmark_std):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        landmark : (T, 2, 68)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)
        feature_masked = torch.zeros_like(feature)  # 時間周波数マスク用の変数

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

            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            # 再生速度変更
            if self.cfg.train.use_time_augment:
                lip, feature, feat_add, landmark, data_len, wav = self.time_augment(lip, feature, feat_add, landmark, wav, upsample, data_len)

            # save_path = Path("~/lip2sp_pytorch/check/lip_time_masking").expanduser()
            # os.makedirs(save_path, exist_ok=True)
            # lip_orig = lip.clone()

            # lip_orig = lip_orig.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
            # lip_orig = lip_orig.to('cpu')
            # print(lip_orig.shape)

            # if self.cfg.model.gray:
            #     lip_orig = lip_orig.expand(-1, -1, -1, 3)

            # torchvision.io.write_video(
            #     filename=str(save_path / "lip_orig.mp4"),
            #     video_array=lip_orig,
            #     fps=self.cfg.model.fps
            # )

            # 動画のマスキング
            if self.cfg.train.use_segment_masking:
                if self.cfg.train.which_seg_mask == "mean":
                    lip = self.segment_masking(lip, lip_mean)
                elif self.cfg.train.which_seg_mask == "seg_mean":
                    lip, landmark = self.segment_masking_segmean(lip, landmark)

            # lip_time_masking = lip.clone()

            # lip_time_masking = lip_time_masking.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
            # lip_time_masking = lip_time_masking.to('cpu')
            # print(lip_time_masking.shape)

            # if self.cfg.model.gray:
            #     lip_time_masking = lip_time_masking.expand(-1, -1, -1, 3)

            # torchvision.io.write_video(
            #     filename=str(save_path / "lip_time_masking.mp4"),
            #     video_array=lip_time_masking,
            #     fps=self.cfg.model.fps
            # )
            # print("time masking ok")
            
        else:
            if lip.shape[1] == 56:
                lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
                lip = self.random_crop(lip, center=True)
                lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
        
        # 標準化
        lip, feature, feat_add, landmark = self.normalization(
            lip, feature, feat_add, landmark, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, landmark_mean, landmark_std,
        )

        if self.train_val_test == "train":
            if self.cfg.train.use_spatial_masking:
                lip = lip.permute(3, 0, 1, 2)
                lip = self.spatial_masking(lip)
                lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
            
            if self.cfg.train.use_time_frequency_masking:
                feature_masked = self.time_frequency_masking(feature)

        # 顔の一部隠す実験で使用
        # lip = lip.permute(3, 0, 1, 2)
        # lip = self.spatial_masking(lip)
        # lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        # 口唇動画の動的特徴量の計算
        if self.cfg.model.delta:
            lip = self.calc_delta(lip)

        # mulaw量子化
        wav = wav.numpy()
        wav_q = mulaw_quantize(wav)
        wav_q = torch.from_numpy(wav_q)
        wav = torch.from_numpy(wav)

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feat_add = feat_add.to(torch.float32)
        landmark = landmark.to(torch.float32)
        feature_masked = feature_masked.to(torch.float32)
    
        return wav, wav_q, lip, feature, feat_add, landmark, feature_masked, data_len            


def collate_time_adjust(batch, cfg):
    """
    フレーム数の調整を行う
    wav, wav_q : (T,)
    lip : (C, H, W, T)
    feature, feat_add : (C, T)
    landmark : (T, 2, 68)
    """
    wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label = list(zip(*batch))

    wav_adjusted = []
    wav_q_adjusted = []
    lip_adjusted = []
    feature_adjusted = []
    feat_add_adjusted = []
    landmark_adjustd = []
    feature_masked_adjusted = []

    # configで指定した範囲でフレーム数を決定
    lip_len = cfg.model.n_lip_frames
    upsample_scale = upsample[0].item()
    feature_len = int(lip_len * upsample_scale)
    wav_len = int(feature_len * cfg.model.hop_length)

    for w, w_q, l, f, f_add, lm, f_masked, d_len in zip(wav, wav_q, lip, feature, feat_add, landmark, feature_masked, data_len):
        # 揃えるlenよりも短い時は足りない分をゼロパディング
        if d_len <= feature_len:
            w_padded = torch.zeros(wav_len)
            w_q_padded = torch.zeros(wav_len) + cfg.model.mulaw_ignore_idx
            w_q_padded = w_q_padded.to(torch.int64)
            l_padded = torch.zeros(l.shape[0], l.shape[1], l.shape[2], lip_len)
            f_padded = torch.zeros(f.shape[0], feature_len)
            f_add_padded = torch.zeros(f_add.shape[0], feature_len)
            lm_padded = torch.zeros(lip_len, lm.shape[1], lm.shape[2])
            f_masked_padded = torch.zeros_like(f_padded)

            # 音響特徴量の系列長をベースに判定しているので、稀に波形のサンプル数が多い場合がある
            # その際に余ったサンプルを除外する（シフト幅的に余りが生じているのでそれを省いている）
            w = w[:wav_len]     
            w_q = w_q[:wav_len]

            w_padded[:w.shape[0]] = w
            w_q_padded[:w.shape[0]] = w_q
            l_padded[..., :l.shape[-1]] = l
            f_padded[:, :f.shape[-1]] = f
            f_add_padded[:, :f_add.shape[-1]] = f_add
            lm_padded[:lm.shape[0], ...] = lm
            f_masked_padded[:, :f_masked.shape[-1]] = f_masked

            w = w_padded
            w_q = w_q_padded
            l = l_padded
            f = f_padded
            f_add = f_add_padded
            lm = lm_padded
            f_masked = f_masked_padded

        # 揃えるlenよりも長い時はランダムに切り取り
        else:
            lip_start_frame = torch.randint(0, l.shape[-1] - lip_len, (1,)).item()
            feature_start_frame = int(lip_start_frame * upsample_scale)
            wav_start_sample = int(feature_start_frame * cfg.model.hop_length)

            w = w[wav_start_sample:wav_start_sample + wav_len]
            w_q = w_q[wav_start_sample:wav_start_sample + wav_len]
            l = l[..., lip_start_frame:lip_start_frame + lip_len]
            f = f[:, feature_start_frame:feature_start_frame + feature_len]
            f_add = f_add[:, feature_start_frame:feature_start_frame + feature_len]
            lm = lm[lip_start_frame:lip_start_frame + lip_len, ...]
            f_masked = f_masked[:, feature_start_frame:feature_start_frame + feature_len]

        assert w.shape[0] == wav_len
        assert w_q.shape[0] == wav_len
        assert l.shape[-1] == lip_len
        assert f.shape[-1] == feature_len
        assert f_add.shape[-1] == feature_len
        assert lm.shape[0] == lip_len
        assert f_masked.shape[-1] == feature_len

        wav_adjusted.append(w)
        wav_q_adjusted.append(w_q)
        lip_adjusted.append(l)
        feature_adjusted.append(f)
        feat_add_adjusted.append(f_add)
        landmark_adjustd.append(lm)
        feature_masked_adjusted.append(f_masked)

    wav = torch.stack(wav_adjusted)
    wav_q = torch.stack(wav_q_adjusted)
    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    feat_add = torch.stack(feat_add_adjusted)
    landmark = torch.stack(landmark_adjustd)
    feature_masked = torch.stack(feature_masked_adjusted)
    data_len = torch.stack(data_len)
    speaker = torch.stack(speaker)

    return wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label

