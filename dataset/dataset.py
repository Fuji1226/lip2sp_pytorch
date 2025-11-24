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
from collections import defaultdict

from dataset.utils import (
    get_spk_emb, 
    get_spk_emb_jvs,
    get_spk_emb_tcd_timit,
    get_spk_emb_vctk,
    get_spk_emb_hifi_captain,
)
from data_process.transform import load_data, load_data_MF


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
        #self.embs.update(get_spk_emb_tcd_timit(cfg))
        self.embs.update(get_spk_emb_jvs(cfg))
        #self.embs.update(get_spk_emb_vctk(cfg))
        #self.embs.update(get_spk_emb_hifi_captain(cfg))

        lip_mean = np.array([cfg.model.avhubert_lip_mean])
        lip_std = np.array([cfg.model.avhubert_lip_std])
        feat_mean_var_std = np.load(str(Path(cfg.train.jvs.stat_path).expanduser()))#!jvsコーパスのみのものに変更
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
            feat_mean=self.feat_mean, #!ないって言われてる、エラー元
            feat_std=self.feat_std #!どうしようかなって感じ
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


class DatasetWithExternalDataRawRE(Dataset):
    def __init__(
        self,
        data_path,
        transform,
        cfg,
        use_datasets=None  # ← 追加
    ):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg

        if use_datasets is None:
            use_datasets = ["default"]

        self.embs = defaultdict(lambda: np.zeros(256))#ダミーの0ベクトル辞書を返す
        if "default" in use_datasets:
            self.embs.update(get_spk_emb(cfg))
        if "tcd_timit" in use_datasets:
            self.embs.update(get_spk_emb_tcd_timit(cfg))
        if "jvs" in use_datasets:
            self.embs.update(get_spk_emb_jvs(cfg))
        if "vctk" in use_datasets:
            self.embs.update(get_spk_emb_vctk(cfg))
        if "hifi_captain" in use_datasets:
            self.embs.update(get_spk_emb_hifi_captain(cfg))

        lip_mean = np.array([cfg.model.avhubert_lip_mean])
        lip_std = np.array([cfg.model.avhubert_lip_std])
        if "vctk" in use_datasets:
            feat_mean_var_std = np.load(str(Path(cfg.train.vctk.stat_path).expanduser()))
            feat_mean = feat_mean_var_std['feat_mean']
            feat_std = feat_mean_var_std['feat_std']
        else:
            # VCTKが使えないとき用にダミー値を設定（プロジェクトに応じて調整）
            feat_mean = np.zeros((80,), dtype=np.float32)  # 例: 80次元の特徴
            feat_std = np.ones((80,), dtype=np.float32)

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
        emo_emb = self.data_path[index]['emotion']#?スカラーを読み込んでいるはず

        # 使わないので適当に
        speaker_idx = torch.tensor(0)
        lang_id = torch.tensor(0)
        is_video = torch.tensor(0)

        wav, feature, feature_avhubert, lip = load_data(audio_path, video_path, self.cfg)#!メルを作ってる(fftしてる)のはここ
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
            emo_emb,
            feature_len,
            lip_len,
            speaker,
            speaker_idx,
            filename,
            lang_id,
            is_video,
        )

class DatasetWithExternalDataRawMF(Dataset):
    def __init__(
        self,
        data_path,
        transform,
        cfg,
        use_datasets=None  # ← 追加
    ):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.cfg = cfg

        if use_datasets is None:
            use_datasets = ["default"]

        self.embs = defaultdict(lambda: np.zeros(256))#ダミーの0ベクトル辞書を返す
        if "default" in use_datasets:
            self.embs.update(get_spk_emb(cfg))
        if "tcd_timit" in use_datasets:
            self.embs.update(get_spk_emb_tcd_timit(cfg))
        if "jvs" in use_datasets:
            self.embs.update(get_spk_emb_jvs(cfg))
        if "vctk" in use_datasets:
            self.embs.update(get_spk_emb_vctk(cfg))
        if "hifi_captain" in use_datasets:
            self.embs.update(get_spk_emb_hifi_captain(cfg))

        lip_mean = np.array([cfg.model.avhubert_lip_mean])
        lip_std = np.array([cfg.model.avhubert_lip_std])
        if "vctk" in use_datasets:
            feat_mean_var_std = np.load(str(Path(cfg.train.vctk.stat_path).expanduser()))
            feat_mean = feat_mean_var_std['feat_mean']
            feat_std = feat_mean_var_std['feat_std']
        else:
            # VCTKが使えないとき用にダミー値を設定（プロジェクトに応じて調整）
            feat_mean = np.zeros((80,), dtype=np.float32)  # 例: 80次元の特徴
            feat_std = np.ones((80,), dtype=np.float32)

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
        emo_emb = self.data_path[index]['emotion']#?スカラーを読み込んでいるはず

        # 使わないので適当に
        speaker_idx = torch.tensor(0)
        lang_id = torch.tensor(0)
        is_video = torch.tensor(0)

        wav, feature, feature_half, feature_double, feature_avhubert, lip = load_data_MF(audio_path, video_path, self.cfg)
        wav = torch.from_numpy(wav)
        feature = torch.from_numpy(feature).permute(1, 0)   # (T, C)
        feature_half = torch.from_numpy(feature_half).permute(1, 0)   # (T, C)
        feature_double = torch.from_numpy(feature_double).permute(1, 0)   # (T, C)
        feature_avhubert = torch.from_numpy(feature_avhubert).permute(1, 0)     # (T, C)
        lip = torch.from_numpy(lip).permute(1, 2, 3, 0)     # (C, H, W, T)
        if self.cfg.train.debug:#TODO,shapeの確認
            print(f"Dataset MF: feature shape before transform: {feature.shape}")
            print(f"Dataset MF: feature_half shape before transform: {feature_half.shape}")
            print(f"Dataset MF: feature_double shape before transform: {feature_double.shape}")
            print(f"Dataset MF: feature_avhubert shape before transform: {feature_avhubert.shape}")
            print(f"Dataset MF: lip shape before transform: {lip.shape}")
            breakpoint()

        lip, feature,feature_half, feature_double, feature_avhubert = self.transform(
            lip=lip,
            feature=feature,
            feature_half =feature_half,
            feature_double=feature_double,
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
            feature_half,
            feature_double,
            feature_avhubert,
            spk_emb,
            emo_emb,
            feature_len,
            lip_len,
            speaker,
            speaker_idx,
            filename,
            lang_id,
            is_video,
        )

class TransformWithExternalDataRaw:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.hflip = T.RandomHorizontalFlip(p=0.5)

    def horizontal_flip(self, lip):
        '''
        lip : (T, C, H, W)
        '''
        lip = self.hflip(lip)
        return lip
    
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

    def normalization(
        self,
        lip,
        feature,
        feature_avhubert,
        lip_mean,
        lip_std,
        feat_mean,
        feat_std,
    ):
        """
        lip : (C, H, W, T)
        feature : (C, T)
        feature_avhubert : (T, C)
        lip_mean, lip_std, feat_mean, feat_std : (C,)
        """
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
        lip = lip / 255.0
        lip = (lip - lip_mean) / lip_std
        feature = (feature - feat_mean) / feat_std
        feature_avhubert = F.layer_norm(feature_avhubert, feature_avhubert.shape[1:]).permute(1, 0)     # (C, T)
        return lip, feature, feature_avhubert

    def segment_masking_segmean(self, lip):
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
    
    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats

    def __call__(
        self,
        lip,
        feature,
        feature_avhubert,
        lip_mean,
        lip_std,
        feat_mean,
        feat_std,
    ):
        """
        lip : (C, H, W, T)
        feature : (T, C)
        feature_avhubert : (T, C)
        """
        feature = feature.permute(1, 0)     # (C, T)
        lip = lip.permute(3, 0, 1, 2)   # (T, C, H, W)

        if self.train_val_test == "train":
            if lip.shape[-1] != self.cfg.model.imsize_cropped:
                if self.cfg.train.use_random_crop:
                    lip = self.random_crop(lip, center=False)
                else:
                    lip = self.random_crop(lip, center=True)
                if self.cfg.train.use_horizontal_flip:
                    lip = self.horizontal_flip(lip)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            if self.cfg.train.use_segment_masking:
                lip = self.segment_masking_segmean(lip)
        else:
            if lip.shape[-1] != self.cfg.model.imsize_cropped:
                lip = self.random_crop(lip, center=True)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        feature_avhubert = self.stacker(feature_avhubert, self.cfg.model.reduction_factor)  # (T // reduction_factor, C * reduction_factor)

        lip, feature, feature_avhubert = self.normalization(
            lip=lip,
            feature=feature,
            feature_avhubert=feature_avhubert,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feature_avhubert = feature_avhubert.to(torch.float32)
        return lip, feature, feature_avhubert

class TransformWithExternalDataRawMF:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.hflip = T.RandomHorizontalFlip(p=0.5)

    def horizontal_flip(self, lip):
        '''
        lip : (T, C, H, W)
        '''
        lip = self.hflip(lip)
        return lip
    
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

    def normalization(
        self,
        lip,
        feature,
        feature_half,
        feature_double,
        feature_avhubert,
        lip_mean,
        lip_std,
        feat_mean,
        feat_std,
    ):
        """
        lip : (C, H, W, T)
        feature : (C, T)
        feature_half : (C, T)
        feature_double : (C, T)
        feature_avhubert : (T, C)
        lip_mean, lip_std, feat_mean, feat_std : (C,)
        """
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        feat_mean = feat_mean.unsqueeze(-1)     # (C, 1)
        feat_std = feat_std.unsqueeze(-1)       # (C, 1)
        lip = lip / 255.0
        lip = (lip - lip_mean) / lip_std
        feature = (feature - feat_mean) / feat_std
        feature_half = (feature_half - feat_mean) / feat_std
        feature_double = (feature_double - feat_mean) / feat_std
        feature_avhubert = F.layer_norm(feature_avhubert, feature_avhubert.shape[1:]).permute(1, 0)     # (C, T)
        if self.cfg.train.debug:#TODO,shapeの確認
            print(f"Transform MF: feature shape after normalization: {feature.shape}")
            print(f"Transform MF: feature_half shape after normalization: {feature_half.shape}")
            print(f"Transform MF: feature_double shape after normalization: {feature_double.shape}")
            print(f"Transform MF: feature_avhubert shape after normalization: {feature_avhubert.shape}")
            print(f"Transform MF: lip shape after normalization: {lip.shape}")
            breakpoint()
        return lip, feature, feature_half, feature_double, feature_avhubert

    def segment_masking_segmean(self, lip):
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
    
    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats

    def __call__(
        self,
        lip,
        feature,
        feature_half,
        feature_double,
        feature_avhubert,
        lip_mean,
        lip_std,
        feat_mean,
        feat_std,
    ):
        """
        lip : (C, H, W, T)
        feature : (T, C)
        feature_avhubert : (T, C)
        """
        feature = feature.permute(1, 0)     # (C, T)
        feature_half = feature_half.permute(1, 0)
        feature_double = feature_double.permute(1, 0)
        lip = lip.permute(3, 0, 1, 2)   # (T, C, H, W)

        if self.train_val_test == "train":
            if lip.shape[-1] != self.cfg.model.imsize_cropped:
                if self.cfg.train.use_random_crop:
                    lip = self.random_crop(lip, center=False)
                else:
                    lip = self.random_crop(lip, center=True)
                if self.cfg.train.use_horizontal_flip:
                    lip = self.horizontal_flip(lip)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

            if self.cfg.train.use_segment_masking:
                lip = self.segment_masking_segmean(lip)
        else:
            if lip.shape[-1] != self.cfg.model.imsize_cropped:
                lip = self.random_crop(lip, center=True)
            lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        feature_avhubert = self.stacker(feature_avhubert, self.cfg.model.reduction_factor)  # (T // reduction_factor, C * reduction_factor)

        lip, feature,feature_half, feature_double, feature_avhubert = self.normalization(
            lip=lip,
            feature=feature,
            feature_half=feature_half,
            feature_double=feature_double,
            feature_avhubert=feature_avhubert,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        lip = lip.to(torch.float32)
        feature = feature.to(torch.float32)
        feature_half = feature_half.to(torch.float32)
        feature_double = feature_double.to(torch.float32)
        feature_avhubert = feature_avhubert.to(torch.float32)
        if self.cfg.train.debug:#TODO,shapeの確認
            print(f"Transform MF: feature shape after all processing: {feature.shape}")
            print(f"Transform MF: feature_half shape after all processing: {feature_half.shape}")
            print(f"Transform MF: feature_double shape after all processing: {feature_double.shape}")
            print(f"Transform MF: feature_avhubert shape after all processing: {feature_avhubert.shape}")
            print(f"Transform MF: lip shape after all processing: {lip.shape}")
            breakpoint()
        return lip, feature, feature_half, feature_double, feature_avhubert