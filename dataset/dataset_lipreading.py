import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pyopenjtalk
import pandas as pd

from dataset.dataset_npz import KablabTransform
from dataset.utils import get_speaker_idx, get_stat_load_data, calc_mean_var_std, get_utt
from data_process.phoneme_encode import classes2index_tts


class LipReadingDataset(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.classes_index = classes2index_tts()

        self.speaker_idx = get_speaker_idx(data_path)

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

        self.path_text_pair_list = get_utt(data_path)

        print(f"n = {self.__len__()}")

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path, text = self.path_text_pair_list[index]
        speaker = data_path.parents[1].name
        label = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        landmark = torch.from_numpy(npz_key['landmark'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        lip, feature, feat_add, text, data_len = self.transform(
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
            text=text,
            classes_index=self.classes_index,
        )
        lip_len = torch.tensor(lip.shape[-1])
        text_len = torch.tensor(text.shape[0] - 1)
        
        return wav, lip, feature, feat_add, text, data_len, text_len, speaker, label


class LipReadingTransform(KablabTransform):
    def __init__(self, cfg, train_val_test):
        super().__init__(cfg, train_val_test)

    def adjust_phoneme_length(self, lip, feature, feat_add, data_len, upsample, phoneme, duration):
        """
        音素ラベルの数を調整し,それに合わせて口唇動画と音響特徴量もフレーム数を調整
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        """
        if len(phoneme) > self.cfg.model.phoneme_length:
            idx = torch.randint(0, len(phoneme) - self.cfg.model.phoneme_length, (1,)).item()
            phoneme = phoneme[idx:idx + self.cfg.model.phoneme_length]
            duration = duration[idx:idx + self.cfg.model.phoneme_length]
            start_time = float(duration[0][0])
            end_time = float(duration[-1][1])

            lip_each_frame_time = torch.arange(lip.shape[-1]) / self.cfg.model.fps

            _, lip_start_frame_index = torch.abs((lip_each_frame_time - start_time)).min(dim=-1)
            _, lip_end_frame_index = torch.abs((lip_each_frame_time - end_time)).min(dim=-1)

            feature_start_frame_index = lip_start_frame_index * upsample
            feature_end_frame_index = lip_end_frame_index * upsample

            lip = lip[..., lip_start_frame_index:lip_end_frame_index]
            feature = feature[:, feature_start_frame_index:feature_end_frame_index]
            feat_add = feat_add[:, feature_start_frame_index:feature_end_frame_index]
            assert int(lip.shape[-1] * upsample) == feature.shape[-1]
            assert int(lip.shape[-1] * upsample) == feat_add.shape[-1]

            data_len = torch.tensor(feature.shape[-1])

        return lip, feature, feat_add, phoneme, data_len

    def text2index(self, text, classes_index):
        """
        音素ラベルを数値列に変換
        """
        text = text.split(" ")
        text.insert(0, "sos")
        text.append("eos")
        phoneme_index = [classes_index[p] if p in classes_index.keys() else None for p in text]
        assert (None in phoneme_index) is False
        return torch.tensor(phoneme_index)

    def __call__(
        self, lip, feature, feat_add, landmark, upsample, data_len, lip_mean, lip_std, feat_mean, feat_std,
        feat_add_mean, feat_add_std, landmark_mean, landmark_std, text, classes_index):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        landmark : (T, 2, 68)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)

        # data augmentation
        if self.train_val_test == "train":
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = super().apply_lip_trans(lip)

            if lip.shape[-1] == 56:
                if self.cfg.train.use_random_crop:
                    lip = super().random_crop(lip, center=False)
                else:
                    lip = super().random_crop(lip, center=True)

            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            # 再生速度変更
            if self.cfg.train.use_time_augment:
                lip, feature, feat_add, landmark, data_len, wav = super().time_augment(lip, feature, feat_add, landmark, wav, upsample, data_len)

            # 動画のマスキング
            if self.cfg.train.use_segment_masking:
                if self.cfg.train.which_seg_mask == "mean":
                    lip = super().segment_masking(lip, lip_mean)
                elif self.cfg.train.which_seg_mask == "seg_mean":
                    lip, landmark = super().segment_masking_segmean(lip, landmark)
        else:
            if lip.shape[1] == 56:
                lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
                lip = super().random_crop(lip, center=True)
                lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        # 標準化
        lip, feature, feat_add, landmark = super().normalization(
            lip, feature, feat_add, landmark, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, landmark_mean, landmark_std,
        )

        # 音素の系列量を調整
        # if self.train_val_test == "train" or self.train_val_test == "val":
        #     lip, feature, feat_add, phoneme, data_len = self.adjust_phoneme_length(lip, feature, feat_add, data_len, upsample, phoneme, duration)

        text = pyopenjtalk.g2p(text)
        text = self.text2index(text, classes_index)

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feat_add = feat_add.to(torch.float32)
        return lip, feature, feat_add, text, data_len


def adjust_max_data_len(data):
    """
    minibatchの中で最大のdata_lenに合わせて0パディングする
    data : (..., T)
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

        new_data.append(d_padded)
    
    return new_data


def collate_time_adjust_lipreading(batch, cfg):
    wav, lip, feature, feat_add, text, data_len, text_len, speaker, label = list(zip(*batch))

    wav = adjust_max_data_len(wav)
    lip = adjust_max_data_len(lip)
    feature = adjust_max_data_len(feature)
    feat_add = adjust_max_data_len(feat_add)
    text = adjust_max_data_len(text)
    
    wav = torch.stack(wav)
    lip = torch.stack(lip)
    feature = torch.stack(feature)
    feat_add = torch.stack(feat_add)
    text = torch.stack(text)
    data_len = torch.stack(data_len)
    text_len = torch.stack(text_len)

    return wav, lip, feature, feat_add, text, data_len, text_len, speaker, label


def collate_time_adjust_ctc(batch):
    wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = list(zip(*batch))

    wav = adjust_max_data_len(wav)
    lip = adjust_max_data_len(lip)
    feature = adjust_max_data_len(feature)
    feat_add = adjust_max_data_len(feat_add)

    wav = torch.stack(wav)
    lip = torch.stack(lip)
    feature = torch.stack(feature)
    feat_add = torch.stack(feat_add)
    data_len = torch.stack(data_len)

    phoneme_len = torch.tensor([p.shape[0] for p in phoneme_index])
    phoneme_index = torch.cat(list(phoneme_index), dim=0)

    return wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label, phoneme_len

