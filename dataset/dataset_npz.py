import os
import sys
import re
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import pickle
import random
from tqdm import tqdm
import numpy as np
import pyopenjtalk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

from dataset.utils import get_speaker_idx, get_stat_load_data, calc_mean_var_std, get_spk_emb, get_utt, adjust_max_data_len
from data_process.phoneme_encode import classes2index_tts, pp_symbols


class KablabDataset(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg
        self.class_to_id, self.id_to_class = classes2index_tts()
        
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

        self.lip_mean = torch.from_numpy(lip_mean)
        self.lip_std = torch.from_numpy(lip_std)
        self.feat_mean = torch.from_numpy(feat_mean)
        self.feat_std = torch.from_numpy(feat_std)
        
        self.path_text_pair_list = get_utt(data_path)

        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path, text = self.path_text_pair_list[index]
        speaker = data_path.parents[1].name
        speaker_idx = torch.tensor(self.speaker_idx[speaker])
        spk_emb = torch.from_numpy(self.embs[speaker])
        label = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])

        lip, feature, text = self.transform(
            lip=lip,
            feature=feature, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            text=text,
            class_to_id=self.class_to_id,
        )
        
        feature_len = torch.tensor(feature.shape[-1])
        lip_len = torch.tensor(lip.shape[-1])
        text_len = torch.tensor(text.shape[0])
        stop_token = torch.zeros(feature_len)
        stop_token[-2:] = 1.0
        return wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label


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
        if center:
            top = left = (self.cfg.model.imsize - self.cfg.model.imsize_cropped) // 2
        else:
            top = torch.randint(0, self.cfg.model.imsize - self.cfg.model.imsize_cropped, (1,))
            left = torch.randint(0, self.cfg.model.imsize - self.cfg.model.imsize_cropped, (1,))
        height = width = self.cfg.model.imsize_cropped
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
            
            for i in mask_idx:
                lip_aug[..., i] = 0
            
            lip_aug = fold(lip_aug).to(input_type)

        elif self.cfg.train.which_spatial_mask == "cutout":
            do_or_through = random.randint(0, 1)
            if do_or_through == 1:
                mask = torch.zeros(H, W)
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
        self, lip, feature, lip_mean, lip_std, feat_mean, feat_std):
        """
        標準化
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        landmark : (T, 2, 68)
        
        lip_mean, lip_std : (C, H, W) or (C,)
        feat_mean, feat_std, feat_add_mean, feat_add_std : (C,)
        landmark_mean, landmark_std : (2,)
        """
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
        lip = (lip - lip_mean) / lip_std        
        feature = (feature - feat_mean) / feat_std
        return lip, feature

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

    def time_augment(self, lip, feature, interp_mode="nearest"):
        """
        再生速度を変更する
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        landmark : (T, 2, 68)
        wav : (T,)
        """
        # 変更する割合を決定
        rate = torch.randint(100 - self.cfg.train.time_augment_rate, 100 + self.cfg.train.time_augment_rate, (1,)) / 100
        
        lip = F.interpolate(
            lip, size=(lip.shape[2], lip.shape[3] * rate), mode=interp_mode, recompute_scale_factor=False)
        feature = feature.unsqueeze(0)      # (1, C, T)
        feat_add = feat_add.unsqueeze(0)    # (1, C, T)
        feature = F.interpolate(feature, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0)    # (C, T)
        feat_add = F.interpolate(feat_add, scale_factor=rate, mode=interp_mode, recompute_scale_factor=False).squeeze(0)  # (C, T)
        
        upsample = 1000 // self.cfg.model.frame_period // self.cfg.model.fps
        data_len = torch.tensor(min(int(lip.shape[-1] * upsample), feature.shape[-1])).to(torch.int)
        lip = lip[..., :data_len // upsample]
        feature = feature[..., :data_len]

        return lip, feature

    def segment_masking(self, lip):
        """
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
    
    def text2index(self, text, class_to_id):
        """
        音素を数値に変換
        """
        text = pyopenjtalk.extract_fullcontext(text)
        text = pp_symbols(text)
        text = [class_to_id[t] for t in text]
        
        # text = pyopenjtalk.g2p(text)
        # text = text.split(" ")
        # text.insert(0, "sos")
        # text.append("eos")
        # text_index = [class_to_id[t] if t in class_to_id.keys() else None for t in text]
        # assert (None in text_index) is False
        return torch.tensor(text)

    def __call__(
        self, lip, feature, lip_mean, lip_std, feat_mean, feat_std, text, class_to_id):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        landmark : (T, 2, 68)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        lip = lip.permute(-1, 0, 1, 2)  # (T, C, H, W)

        # data augmentation
        if self.train_val_test == "train":
            # 見た目変換
            lip = self.apply_lip_trans(lip)

            if lip.shape[-1] == self.cfg.model.imsize:
                if self.cfg.train.use_random_crop:
                    lip = self.random_crop(lip, center=False)
                else:
                    lip = self.random_crop(lip, center=True)

            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            # 再生速度変更
            if self.cfg.train.use_time_augment:
                lip, feature = self.time_augment(lip, feature)

            # time masking
            if self.cfg.train.use_segment_masking:
                lip = self.segment_masking(lip)
            
        else:
            if lip.shape[-1] == self.cfg.model.imsize:
                lip = self.random_crop(lip, center=True)
                lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)
        
        # 標準化
        lip, feature = self.normalization(lip, feature, lip_mean, lip_std, feat_mean, feat_std)

        if self.train_val_test == "train":
            if self.cfg.train.use_spatial_masking:
                lip = lip.permute(3, 0, 1, 2)   # (T, C, H, W)
                lip = self.spatial_masking(lip)
                lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        text = self.text2index(text, class_to_id)
        return lip, feature, text


def collate_time_adjust(batch, cfg):
    wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = list(zip(*batch))

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
        
    text = adjust_max_data_len(text)
    stop_token = adjust_max_data_len(stop_token)
    
    wav = torch.stack(wav_adjusted)
    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    text = torch.stack(text)
    stop_token = torch.stack(stop_token)
    spk_emb = torch.stack(spk_emb)
    feature_len = torch.stack(feature_len)
    lip_len = torch.stack(lip_len)
    text_len = torch.stack(text_len)
    speaker_idx = torch.stack(speaker_idx)
    return wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label


def collate_time_adjust_tts(batch, cfg):
    wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = list(zip(*batch))
    
    wav = adjust_max_data_len(wav)
    lip = adjust_max_data_len(lip)
    feature = adjust_max_data_len(feature)
    text = adjust_max_data_len(text)
    stop_token = adjust_max_data_len(stop_token)
    
    wav = torch.stack(wav)
    lip = torch.stack(lip)
    feature = torch.stack(feature)
    text = torch.stack(text)
    stop_token = torch.stack(stop_token)
    spk_emb = torch.stack(spk_emb)
    feature_len = torch.stack(feature_len)
    lip_len = torch.stack(lip_len)
    text_len = torch.stack(text_len)
    speaker_idx = torch.stack(speaker_idx)
    return wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label