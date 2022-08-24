import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/data_process").expanduser()))

import numpy as np
import skvideo.io
import librosa
import pyrubberband
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from .dataset_npz import load_mean_std

try:
    from ..data_process.transform_no_chainer import calc_sp, calc_feat_add
except:
    from data_process.transform_no_chainer import calc_sp, calc_feat_add


class Lip2SPDataset(Dataset):
    def __init__(self, data_path, mean_std_path, transform, cfg, test):
        super().__init__()
        self.data_path = data_path
        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, cfg.model.name, test)
        self.lip_mean = self.lip_mean.to('cpu').detach().numpy().copy()
        self.lip_std = self.lip_std.to('cpu').detach().numpy().copy()
        self.feat_mean = self.feat_mean.to('cpu').detach().numpy().copy()
        self.feat_std = self.feat_std.to('cpu').detach().numpy().copy()
        self.feat_add_mean = self.feat_add_mean.to('cpu').detach().numpy().copy()
        self.feat_add_std = self.feat_add_std.to('cpu').detach().numpy().copy()
        self.transform = transform
        self.cfg = cfg
        self.test = test

        print(f"n = {self.__len__()}")

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        (video_path, audio_path) = self.data_path[idx]
        video_path, audio_path = Path(video_path), Path(audio_path)
        speaker = audio_path.parents[0].name
        label = audio_path.stem

        lip = skvideo.io.vread(str(video_path), outputdict={"-s": "48x48"})     # (T, H, W, C)
        wav, fs = librosa.load(str(audio_path), sr=16000)     # (T,)
        wav = wav / np.max(np.abs(wav), axis=0)     # [-1, 1]に正規化

        wav, lip, feature, feat_add, data_len = self.transform(
            lip=lip, 
            wav=wav, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            feat_add_mean=self.feat_add_mean, 
            feat_add_std=self.feat_add_std,
        )

        return wav, lip, feature, feat_add, data_len, speaker, label


class Lip2SPTransform:
    def __init__(self, cfg, train_val_test, upsample_scale=2):
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.upsample_scale = upsample_scale

        self.color_jitter = T.ColorJitter(brightness=[0.5, 1.5], contrast=0, saturation=1, hue=0.2)
        self.blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)) # ぼかし
        self.horizontal_flip = T.RandomHorizontalFlip(p=0.5)  # 左右反転
        self.rotation = T.RandomRotation(degrees=(0, 10))     # degressの範囲で回転
        self.pad = T.RandomCrop(size=(48, 48), padding=4)

    def change_speed(self, lip, wav, rate):
        """
        lip : (T, H, W, C)
        wav : (T,)
        rate : 速さを何倍にするか。0.8とか1.2とかそんな感じです。
        """
        delay_rate = 100
        fast_rate = int(rate * 100)
        delay_out = []
        fast_out = []

        # 一度フレームを複製
        for i in range(lip.shape[0]):
            for j in range(delay_rate):
                delay_out.append(lip[i, ...])
        print(f"rate = {rate}, delay_rate = {delay_rate}, fast_rate = {fast_rate}")
        print(f"{len(delay_out)}")
        # 増やしたフレームから間引いて速さをrateで指定したものにする
        for i in range(len(delay_out) // fast_rate):
            fast_out.append(delay_out[int(i * fast_rate)])

        lip = np.array(fast_out)

        # time stretchで音声の速度も変更
        wav = pyrubberband.pyrb.time_stretch(wav, self.cfg.model.sampling_rate, rate=fast_rate / delay_rate)
        return lip, wav

    def apply_lip_trans(self, lip):
        """
        口唇動画へのデータ拡張
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
            lip = self.roration(lip)
        return lip

    def normalization(self, lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        標準化
        """
        # 口唇動画の平均、標準偏差を時間方向のみ求めた場合
        if lip_mean.ndim == 3:
            lip_mean = lip_mean[None, ...]
            lip_std = lip_std[None, ...]
        # 時間、空間両方で求めた場合
        elif lip_mean.ndim == 1:
            lip_mean = lip_mean[None, :, None, None]
            lip_std = lip_std[None, :, None, None]

        feat_mean = feat_mean[None, :]
        feat_std = feat_std[None, :]
        feat_add_mean = feat_add_mean[None, :]
        feat_add_std = feat_add_std[None, :]

        lip = (lip.astype(np.float32) - lip_mean) / lip_std
        feature = (feature - feat_mean) / feat_std
        feat_add = (feat_add - feat_add_mean) / feat_add_std

        return lip, feature, feat_add

    def calc_delta(self, lip):
        """
        口唇動画の動的特徴量の計算
        """
        if self.cfg.model.delta:
            lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]
            lip_pad = lip_pad.astype(lip.dtype)
            lip_pad = gaussian_filter(
                lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)
            lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
            lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
            lip_acc = lip_pad[..., 0:-2] + \
                lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
            lip = np.vstack((lip, lip_diff, lip_acc))
        return lip

    def __call__(self, lip, wav, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        if self.train_val_test == "train":
            # 動画、音声のスピードを変換
            if self.cfg.train.use_change_speed:
                print("\nchange speed")
                judge = np.random.randint(0, 3, (1,)).item()
                if judge == 0:
                    rate = self.cfg.train.change_speed_low
                elif judge == 1:
                    rate = self.cfg.train.change_speed_high
                else:
                    rate = 1
                print(f"\nrate = {rate}")
                print(f"before : {lip.shape}, {wav.shape}")
                lip, wav = self.change_speed(lip, wav, rate)
                print(f"after : {lip.shape}, {wav.shape}")

            # 口唇動画へのデータ拡張
            lip = torch.from_numpy(lip).permute(0, -1, 1, 2)    # (T, C, H, W)
            if self.cfg.train.use_lip_transform:
                print("lip transform")
                lip = self.apply_lip_trans(lip)
            lip = lip.to('cpu').detach().numpy().copy()
        else:
            lip = torch.from_numpy(lip).permute(0, -1, 1, 2)    # (T, C, H, W)
            lip = lip.to('cpu').detach().numpy().copy()

        feature = calc_sp(wav, self.cfg)    # (T, C)
        feat_add, T = calc_feat_add(wav, feature, self.cfg)
        feature = feature[:T, :]
        if self.cfg.model.name == "mspec80":
            assert feature.shape[-1] == 80
        elif self.cfg.model.name == "world_melfb":
            assert feature.shape[-1] == 32

        # 多少強引にデータの長さを揃える
        data_len = min(feature.shape[0] // self.upsample_scale * self.upsample_scale,  lip.shape[0] * self.upsample_scale)
        lip = lip[:data_len // self.upsample_scale, ...]
        feature = feature[:data_len, :]
        feat_add = feat_add[:data_len, :]

        # 標準化
        lip, feature, feat_add = self.normalization(
            lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std
        )
        lip = lip.transpose(1, 2, 3, 0)     # (C, H, W, T)
        feature = feature.transpose(1, 0)   # (C, T)
        feat_add = feat_add.transpose(1, 0) # (C, T)

        # 口唇動画の動的特徴量の計算
        lip = self.calc_delta(lip)

        wav = torch.from_numpy(wav).to(torch.float)
        lip = torch.from_numpy(lip).to(torch.float)
        feature = torch.from_numpy(feature).to(torch.float)
        feat_add = torch.from_numpy(feat_add).to(torch.float)
        data_len = torch.tensor(data_len)
        return wav, lip, feature, feat_add, data_len


def collate_time_adjust(batch, cfg):
    """
    フレーム数の調整を行う
    """
    wav, lip, feature, feat_add, data_len, speaker, label = list(zip(*batch))

    lip_adjusted = []
    feature_adjusted = []
    feat_add_adjusted = []

    # configで指定した範囲でフレーム数を決定
    lip_len = torch.randint(cfg.model.lip_min_frame, cfg.model.lip_max_frame, (1,)).item()
    upsample_scale = 2      # 音響特徴量は口唇動画1フレームに対して2フレームあるので2倍
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

    return lip, feature, feat_add, data_len, speaker, label