from cProfile import label
import os
from pathlib import Path
import sys

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import Dataset
import numpy as np
import librosa

from dataset.dataset_npz import load_mean_std
from data_process.feature import wave2mel
from data_process.cut_silence import SoxEffects
from data_process.mulaw import mulaw_quantize


class WaveNetDataset(Dataset):
    def __init__(self, data_path, mean_std_path, transform, cfg, test):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.test = test

        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, cfg.model.name, test)

        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path = self.data_path[index]
        label = Path(data_path).stem

        npz_key = np.load(data_path)
        wav = torch.from_numpy(npz_key["wav"]).clone()

        wav, feature = self.transform(wav, self.feat_mean, self.feat_std)
        return wav, feature, label
        

class WaveNetTransform:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.cut_silence = SoxEffects(
            sample_rate=self.cfg.model.sampling_rate,
            sil_threshold=0.3,
            sil_duration=0.1,
        )

    def normalization(self, feature, feat_mean, feat_std):
        feat_mean = feat_mean.to('cpu').detach().numpy().copy()
        feat_std = feat_std.to('cpu').detach().numpy().copy()
        feat_mean = feat_mean[:, None]
        feat_std = feat_std[:, None]
        feature = (feature - feat_mean) / feat_std
        return feature

    def time_adjust(self, wav):
        idx = np.random.randint(0, wav.shape[0] - self.cfg.model.n_samples, (1,))
        wav = wav[int(idx):int(idx + self.cfg.model.n_samples)]
        return wav

    def __call__(self, wav, feat_mean, feat_std):
        wav = wav[None, :]  # (1, T)

        # 無音区間の切り取り
        if self.train_val_test == "train" or self.train_val_test == "val":
            wav = self.cut_silence(wav, self.cfg.model.sampling_rate)

        wav = wav.squeeze(0)    # (T,)
        wav = wav.to('cpu').detach().numpy().copy()

        # サンプル数調整
        if self.train_val_test == "train" or self.train_val_test == "val":
            wav = self.time_adjust(wav)

        # 条件付けに使用する音響特徴量の計算
        feature = wave2mel(
            wave=wav,
            fs=self.cfg.model.sampling_rate,
            frame_period=self.cfg.model.frame_period,
            n_mels=self.cfg.model.n_mel_channels,
            fmin=self.cfg.model.f_min,
            fmax=self.cfg.model.f_max,
        )

        # 標準化
        shift = self.cfg.model.hop_length
        feature = self.normalization(feature, feat_mean, feat_std)
        feature = feature[:, :-1]
        wav = wav[:int(feature.shape[-1] * shift)]

        # mulawアルゴリズムによる量子化
        wav = mulaw_quantize(wav)

        # assert feature.shape[-1] * shift == wav.shape[0]
        return wav, feature


