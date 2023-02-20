from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import pyopenjtalk
from tqdm import tqdm
from torchvision import transforms as T

from data_process.transform import load_data
from dataset.utils import select_data, get_stat_load_data_raw, calc_mean_var_std, get_spk_emb
from data_process.phoneme_encode import classes2index_tts
from dataset.dataset_lipreading import adjust_max_data_len


class DatasetTTSFace(Dataset):
    def __init__(self, data_dir, bbox_dir, landmark_dir, df, train_df, transform, cfg):
        super().__init__()
        self.transform = transform
        self.cfg = cfg
        self.classes_index = classes2index_tts()
        data_path_list = select_data(data_dir, bbox_dir, landmark_dir, df, cfg)
        train_data_path_list = select_data(data_dir, bbox_dir, landmark_dir, train_df, cfg)
        self.data_path_list = data_path_list

        lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list = \
            get_stat_load_data_raw(train_data_path_list, cfg)
        lip_mean, _, lip_std = calc_mean_var_std(lip_mean_list, lip_var_list, lip_len_list)
        feat_mean, _, feat_std = calc_mean_var_std(feat_mean_list, feat_var_list, feat_len_list)
        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)

        self.embs = get_spk_emb(cfg)

        print(f"n = {self.__len__()}")

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        video_path, audio_path, bbox_path, landmark_path, text_path = self.data_path_list[index]
        speaker = video_path.parents[0].name
        label = video_path.stem

        spk_emb = torch.from_numpy(self.embs[speaker])
        
        wav, lip, feature, data_len, text = load_data(video_path, audio_path, bbox_path, landmark_path, text_path, self.cfg)
        wav = torch.from_numpy(wav)
        lip = torch.from_numpy(lip)
        feature = torch.from_numpy(feature)
        data_len = torch.tensor(data_len)

        text, feature, lip = self.transform(
            text=text, 
            classes_index=self.classes_index,
            feature=feature, 
            lip=lip,
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std,
            lip_mean=self.lip_mean,
            lip_std=self.lip_std,
        )
        text_len = torch.tensor(text.shape[0])
        feature_len = torch.tensor(feature.shape[1])
        lip_len = torch.tensor(lip.shape[-1])
        if self.cfg.train.name == "tts":
            stop_token = torch.zeros(feature_len)
        elif self.cfg.train.name == "face_gen_text":
            stop_token = torch.zeros(lip_len)
        stop_token[-self.cfg.train.n_stop_token_frame:] = 1.0
        return wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label


class TransformTTSFace:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test

    def normalization(self, feature, lip, feat_mean, feat_std, lip_mean, lip_std):
        """
        標準化
        feature : (C, T)
        feat_mean, feat_std : (C,)
        lip : (C, H, W, T)
        lip_mean, lip_std : (C,)
        """
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        feature = (feature - feat_mean) / feat_std
        lip = (lip - lip_mean) / lip_std
        return feature, lip

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

    def __call__(self, text, classes_index, feature, lip, feat_mean, feat_std, lip_mean, lip_std):
        """
        text : (T,)
        feature : (T, C)
        lip : (C, H, W, T)
        """
        lip = lip.permute(-1, 0, 1, 2)      # (T, C, H, W)
        lip = self.random_crop(lip, center=True)
        lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        feature = feature.permute(1, 0)     # (C, T)
        feature, lip = self.normalization(feature, lip, feat_mean, feat_std, lip_mean, lip_std)
        feature = feature.to(torch.float32)
        lip = lip.to(torch.float32)

        text = pyopenjtalk.g2p(text)
        text = self.text2index(text, classes_index)
        return text, feature, lip


def collate_time_adjust_tts_face(batch, cfg):
    wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label = list(zip(*batch))

    wav = adjust_max_data_len(wav)
    text = adjust_max_data_len(text)
    feature = adjust_max_data_len(feature)
    lip = adjust_max_data_len(lip)
    stop_token = adjust_max_data_len(stop_token)

    wav = torch.stack(wav)
    text = torch.stack(text)
    feature = torch.stack(feature)
    lip = torch.stack(lip)
    stop_token = torch.stack(stop_token)
    text_len = torch.stack(text_len)
    feature_len = torch.stack(feature_len)
    lip_len = torch.stack(lip_len)
    spk_emb = torch.stack(spk_emb)

    return wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label