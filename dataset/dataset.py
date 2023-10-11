import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import random
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T

from dataset.utils import (
    get_spk_emb, 
    get_spk_emb_hifi_captain,
)
from data_process.transform import load_data


class DatasetWithExternalDataRaw(Dataset):
    def __init__(
        self,
        data_path,
        transform,
        cfg,
    ):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg
        self.embs = get_spk_emb(cfg)
        self.embs.update(get_spk_emb_hifi_captain())

        lip_mean = np.array([cfg.model.avhubert_lip_mean])
        lip_std = np.array([cfg.model.avhubert_lip_std])
        feat_mean_var_std = np.load(str(Path(cfg.train.hifi_captain_feat_mean_var_std_path).expanduser()))
        feat_mean = feat_mean_var_std['feat_mean']
        feat_std = feat_mean_var_std['feat_std']
        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        audio_path = self.data_path[index]['audio_path']
        video_path = self.data_path[index]['video_path']
        speaker = self.data_path[index]['speaker']
        filename = self.data_path[index]['filename']
        spk_emb = torch.from_numpy(self.embs[speaker])

        # 使わないので適当に
        speaker_idx = torch.tensor(0)
        lang_id = torch.tensor(0)
        is_video = torch.tensor(0)

        wav, feature, feature_avhubert, lip = load_data(audio_path, video_path, self.cfg)
        wav = torch.from_numpy(wav)
        feature = torch.from_numpy(feature).permute(1, 0)   # (T, C)
        feature_avhubert = torch.from_numpy(feature_avhubert).permute(1, 0)     # (T, C)
        lip = torch.from_numpy(lip).permute(1, 2, 3, 0)     # (C, H, W, T)
        
        lip, feature, feature_avhubert = self.transform(
            lip=lip, 
            feature=feature, 
            feature_avhubert=feature_avhubert,
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std
        )
        feature_len = torch.tensor(feature.shape[-1])
        lip_len = torch.tensor(lip.shape[-1])

        return (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            feature_len,
            lip_len,
            speaker,
            speaker_idx,
            filename,
            lang_id,
            is_video,
        )