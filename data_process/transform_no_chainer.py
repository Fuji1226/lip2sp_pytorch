"""
データのロード、前処理
make_npz.pyを実行するとここの処理が行われます
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/data_process").expanduser()))

from pathlib import Path
import cv2
import numpy as np
import torch
import librosa
from scipy.interpolate import interp1d
from pysptk import swipe
import torchvision

from utils import get_upsample
from data_process.feature import wave2mel, wav2world


def calc_sp(wave, cfg):
    """
    y : (T, C)
    """
    if cfg.model.feature_type == "mspec":
        # 対数メルスペクトログラム
        y = wave2mel(
            wave, cfg.model.sampling_rate, cfg.model.frame_period, 
            n_mels=cfg.model.n_mel_channels, fmin=cfg.model.f_min, fmax=cfg.model.f_max).T    

    elif cfg.model.feature_type == "world":
        mcep, clf0, vuv, cap, fbin, _ = wav2world(
            wave, cfg.model.sampling_rate, frame_period=cfg.model.frame_period, comp_mode=cfg.model.comp_mode)
        y = np.hstack([mcep, clf0.reshape(-1, 1), vuv.reshape(-1, 1), cap])
    return y


def fill_nan(x):
    flag = np.isfinite(x)
    if flag.all():
        return x

    idx = np.arange(flag.size)
    func = interp1d(
        idx[flag], x[flag],
        bounds_error=False,
        fill_value=(x[flag][0], x[flag][-1])
    )
    x[~flag] = func(idx[~flag])

    assert np.isfinite(x).all()
    return x


def continuous_f0(f0, amin=70):
    f0[f0 < amin] = np.nan
    vuv = np.isfinite(f0)
    if np.isfinite(f0).sum() < 3:
        f0[:] = amin
    f0 = fill_nan(f0)
    return f0, vuv


def load_mp4(path, gray=False):
    """
    可視化が楽だったので、torchvisionを利用して読み込む
    リサイズで画像のピクセル数を変更
    """
    movie = cv2.VideoCapture(str(path))
    fps = int(movie.get(cv2.CAP_PROP_FPS))
    movie.release()

    lip, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")    # lip : (T, W, H, C)
    resizer = torchvision.transforms.Resize((48, 48))
    lip_resize = resizer(lip.permute(0, -1, 1, 2))  # (T, C, W, H)
    lip_resize = lip_resize.permute(1, 2, 3, 0)  # (C, W, H, T)

    return lip_resize, fps


def calc_feat_add_taguchi(wav, feature, cfg):
    hop_length = cfg.model.sampling_rate * cfg.model.frame_period // 1000  
    
    power = librosa.feature.rms(y=wav, frame_length=hop_length*2, hop_length=hop_length).squeeze()
    power = fill_nan(power)
    f0 = swipe(wav.astype("float64"), cfg.model.sampling_rate, hop_length, min=70.0, otype='f0').squeeze()

    f0, vuv = continuous_f0(f0)
    T = min(power.size, f0.size, feature.shape[0])
    
    feat_add = np.vstack((f0[:T], vuv[:T], power[:T])).T
    feat_add = np.log(np.maximum(feat_add, 1.0e-7))
    return feat_add, T


def calc_feat_add(wav, feature, cfg):
    """
    音声のrmsとf0を計算
    """
    # パワーを計算
    power = librosa.feature.rms(y=wav, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
    power = fill_nan(power)
    power = librosa.amplitude_to_db(power, ref=np.max)

    # 基本周波数を計算
    mcep, clf0, vuv, cap, fbin, _ = wav2world(
        wave=wav, fs=cfg.model.sampling_rate, frame_period=cfg.model.frame_period,
    )

    # f0, voiced_flag, voiced_probs = librosa.pyin(
    #     y=wav,
    #     fmin=librosa.note_to_hz('C2'),
    #     fmax=librosa.note_to_hz('C7'),
    #     sr=cfg.model.sampling_rate,
    #     frame_length=cfg.model.win_length,
    #     win_length=cfg.model.win_length // 2,
    #     hop_length=cfg.model.hop_length,
    #     fill_na=None,
    # )

    T = min(power.shape[0], clf0.shape[0], feature.shape[0])
    feat_add = np.vstack((clf0[:T], power[:T])).T   # (T, C)
    return feat_add, T


def load_data_for_npz(video_path, audio_path, cfg):
    """
    lip : (C, H, W, T)
    feature, feat_add : (T, C)
    """
    lip, fps = load_mp4(str(video_path), cfg.model.gray)   # lipはtensor
    wav, fs = librosa.load(str(audio_path), sr=cfg.model.sampling_rate, mono=None)
    wav = wav / np.max(np.abs(wav), axis=0)
    upsample = get_upsample(fps, fs, cfg.model.frame_period)
    
    # 音響特徴量への変換
    feature = calc_sp(wav, cfg)
    feat_add, T = calc_feat_add(wav, feature, cfg)
    feature = feature[:T]

    data_len = min(len(feature) // upsample * upsample,  lip.shape[-1] * upsample)
    feature = feature[:data_len]
    feat_add = feat_add[:data_len]
    lip = lip[..., :data_len // upsample]

    lip = lip.to('cpu').detach().numpy().copy()

    ret = (lip, feature, feat_add, upsample)

    return wav, ret, data_len
