"""
LipReading用のデータセット
基本的には口唇音声変換で使っているものを継承し,音素ラベルのための処理を追加しています
"""
import os
import sys
from pathlib import Path
sys.path.append(Path("~/lip2sp_pytorch/data_process").expanduser())

import numpy as np
import torch
import torch.nn as nn
from .dataset_npz import KablabDataset, KablabTransform
from data_process.phoneme_encode import get_classes, classes2index ,get_phoneme_info


def get_data_simultaneously(data_root, name):
    """
    npzファイルとlabファイルを取得
    """
    data_path = []

    for curdir, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".npz"):
                # mspecかworldかの分岐
                if f"{name}" in Path(file).stem:
                    # npzファイルまでのパス
                    npz_path = os.path.join(curdir, file)
                    # print(f"data_path = {data_path}")

                    # labファイルまでのパスを取得するためにファイルの名前をいじる
                    filename_model_removed = Path(file).stem.split("_")[:-1]
                    # print(f"filename_model_removed = {filename_model_removed}")
                    filename_alignment = "_".join(filename_model_removed)
                    # print(f"filename_alignment = {filename_alignment}")

                    # labファイルまでのパス
                    alignment_path = os.path.join(curdir, f"{filename_alignment}.lab")
                    # print(f"alignment_path = {alignment_path}")
                    
                    # どちらもファイルであることを確認し，unified_pathに追加
                    if os.path.isfile(npz_path) and os.path.isfile(alignment_path):
                        data_path.append([npz_path, alignment_path])

    return data_path 


class LipReadingDataset(KablabDataset):
    def __init__(self, data_path, mean_std_path, transform, cfg, test, classes):
        super().__init__(data_path, mean_std_path, transform, cfg, test)
        self.transform = transform
        self.classes = classes
        self.classes_index = classes2index(self.classes)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        [npz_path, alignment_path] = self.data_path[index]
        speaker = Path(npz_path).parents[0].name
        label = Path(npz_path).stem

        npz_key = np.load(str(npz_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        # 音素ラベルとその継続時間を取得
        phoneme, duration = get_phoneme_info(alignment_path)

        lip, feature, feat_add, phoneme_index, data_len = self.transform(
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
            phoneme=phoneme,
            duration=duration,
            classes_index=self.classes_index,
        )
        
        return wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label


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

    def phoneme2index(self, phoneme, classes_index):
        """
        音素ラベルを数値列に変換
        nn.embeddingを使用した方が良さそうなのでonehotではなくこっちを使用
        """
        # # 最初に"sos"を追加
        # phoneme.insert(0, "sos")

        # # 最後に"eos"を追加
        # phoneme.append("eos")

        # 数値列に変換
        phoneme_index = [classes_index[p] if p in classes_index.keys() else None for p in phoneme]
        assert (None in phoneme_index) is False

        return torch.tensor(phoneme_index)

    def __call__(
        self, lip, feature, feat_add, upsample, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std, 
        phoneme, duration, classes_index
    ):
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)

        # data augmentation
        if self.train_val_test == "train":
            lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)
            lip = super().apply_lip_trans(lip)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        # 標準化
        lip, feature, feat_add = super().normalization(
            lip, feature, feat_add, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std
        )

        # 音素の系列量を調整
        # if self.train_val_test == "train" or self.train_val_test == "val":
        #     lip, feature, feat_add, phoneme, data_len = self.adjust_phoneme_length(lip, feature, feat_add, data_len, upsample, phoneme, duration)

        # 口唇動画の動的特徴量の計算
        lip = super().calc_delta(lip)

        # 音素ラベルをOnehot表現に変換
        phoneme_index = self.phoneme2index(phoneme, classes_index)

        return lip, feature.to(torch.float32), feat_add.to(torch.float32), phoneme_index, data_len


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

        new_data.append(d_padded)
    
    return new_data


def collate_time_adjust(batch):
    wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = list(zip(*batch))

    wav = adjust_max_data_len(wav)
    lip = adjust_max_data_len(lip)
    feature = adjust_max_data_len(feature)
    feat_add = adjust_max_data_len(feat_add)
    phoneme_index = adjust_max_data_len(phoneme_index)
    
    wav = torch.stack(wav)
    lip = torch.stack(lip)
    feature = torch.stack(feature)
    feat_add = torch.stack(feat_add)
    phoneme_index = torch.stack(phoneme_index)
    data_len = torch.stack(data_len)

    return wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label