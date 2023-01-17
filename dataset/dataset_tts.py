from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import pyopenjtalk
from tqdm import tqdm
import librosa

from data_process.feature import wav2mel
from dataset.utils import get_speaker_idx, get_stat_load_data, calc_mean_var_std, get_utt, get_spk_emb
from data_process.phoneme_encode import classes2index_tts
from dataset.dataset_lipreading import adjust_max_data_len


class DatasetTTS(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg
        self.classes_index = classes2index_tts()

        # 話者ID
        self.speaker_idx = get_speaker_idx(data_path)

        # 話者embedding
        self.embs = get_spk_emb(cfg)

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

        self.path_text_pair_list = get_utt(data_path)

        print(f"n = {self.__len__()}")

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path, text = self.path_text_pair_list[index]
        speaker = data_path.parents[1].name

        spk_emb = torch.from_numpy(self.embs[speaker])

        # 話者名を話者IDに変換
        speaker = torch.tensor(self.speaker_idx[speaker])
        label = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        feature = torch.from_numpy(npz_key['feature'])
        data_len = torch.from_numpy(npz_key['data_len'])

        # # 無音区間を除去
        # wav = npz_key["wav"]
        # _, trim_index = librosa.effects.trim(
        #     y=wav,
        #     top_db=30,
        #     frame_length=self.cfg.model.win_length,
        #     hop_length=self.cfg.model.hop_length,
        # )
        # margin = self.cfg.model.sampling_rate // 100
        # wav = wav[trim_index[0] - margin:trim_index[1] + margin]
        # feature = wav2mel(wav, self.cfg, ref_max=False).T   # (T, C)

        # wav = torch.from_numpy(wav)
        # feature = torch.from_numpy(feature)

        # if feature.shape[0] % self.cfg.model.reduction_factor != 0:
        #     feature = feature[:-(feature.shape[0] % self.cfg.model.reduction_factor), :]

        text, feature = self.transform(
            text=text, 
            classes_index=self.classes_index,
            feature=feature, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std,
        )
        text_len = torch.tensor(text.shape[0])
        feature_len = torch.tensor(feature.shape[1])
        stop_token = torch.zeros(feature_len)
        stop_token[-2:] = 1.0
        return wav, text, feature, stop_token, text_len, feature_len, spk_emb, speaker, label


class TransformTTS:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test

    def normalization(self, feature, feat_mean, feat_std):
        """
        標準化
        feature, feat_add : (C, T)
        feat_mean, feat_std, feat_add_mean, feat_add_std : (C,)
        """
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)

        feature = (feature - feat_mean) / feat_std
        return feature

    def text2index(self, text, classes_index):
        """
        音素を数値に変換
        """
        text = text.split(" ")
        text.insert(0, "sos")
        text.append("eos")
        text_index = [classes_index[t] if t in classes_index.keys() else None for t in text]
        assert (None in text_index) is False
        return torch.tensor(text_index)

    def __call__(self, text, classes_index, feature, feat_mean, feat_std):
        """
        text : (T,)
        feature : (T, C)
        """
        feature = feature.permute(1, 0)     # (C, T)
        feature = self.normalization(feature, feat_mean, feat_std)
        feature = feature.to(torch.float32)

        text = pyopenjtalk.g2p(text)
        text = self.text2index(text, classes_index)
        return text, feature


def collate_time_adjust_tts(batch, cfg):
    wav, text, feature, stop_token, text_len, feature_len, spk_emb, speaker, label = list(zip(*batch))

    wav = adjust_max_data_len(wav)
    text = adjust_max_data_len(text)
    feature = adjust_max_data_len(feature)
    stop_token = adjust_max_data_len(stop_token)

    wav = torch.stack(wav)
    text = torch.stack(text)
    feature = torch.stack(feature)
    stop_token = torch.stack(stop_token)
    text_len = torch.stack(text_len)
    feature_len = torch.stack(feature_len)
    spk_emb = torch.stack(spk_emb)

    return wav, text, feature, stop_token, text_len, feature_len, spk_emb, speaker, label