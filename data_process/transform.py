"""
データの処理
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import cv2
import numpy as np
import librosa
from scipy.interpolate import interp1d
from pysptk import swipe
import torchvision
import pandas as pd
import pydub
import torch

from data_process.feature import wav2mel, wav2world
from data_process.face_crop import get_crop_info
from data_process.face_crop_align import FaceAligner, get_landmark


def get_upsample(cfg):
    """
    動画のfpsと音響特徴量のフレームあたりの秒数から対応関係を求める
    """
    n_frames = 1000 // cfg.model.frame_period
    upsample = n_frames // cfg.model.fps
    return int(upsample)


def calc_sp(wav, cfg):
    """
    y : (T, C)
    """
    if cfg.model.name == "mspec80":
        # 対数メルスペクトログラム
        y = wav2mel(wav, cfg, ref_max=False).T

    elif cfg.model.name == "world_melfb":
        # WORLD特徴量
        mcep, clf0, vuv, cap, fbin, _ = wav2world(
            wav, cfg.model.sampling_rate, frame_period=cfg.model.frame_period, cfg=cfg)
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


def load_mp4(path, cfg):
    """
    口唇動画読み込み
    リサイズで画像のピクセル数を変更
    グレースケールへの変更も行う
    """
    movie = cv2.VideoCapture(str(path))
    fps = int(movie.get(cv2.CAP_PROP_FPS))
    movie.release()

    lip, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")    # lip : (T, W, H, C)
    lip_resize = torchvision.transforms.functional.resize(lip.permute(0, -1, 1, 2), [cfg.model.imsize, cfg.model.imsize])   # (T, C, W, H)

    if cfg.model.gray:
        rgb2gray = torchvision.transforms.Grayscale()
        lip_resize = rgb2gray(lip_resize)
        assert lip_resize.shape[1] == 1

    lip_resize = lip_resize.permute(1, 2, 3, 0)  # (C, W, H, T)

    return lip_resize, fps


def load_landmark(path):
    df = pd.read_csv(str(path), header=None)

    landmark_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        landmark_list.append([coords[0::2], coords[1::2]])
    
    return np.array(landmark_list)  # (T, 2, 68)


def calc_feat_add_taguchi(wav, feature, cfg):
    """
    田口さんが使用されていたもの
    f0の推定精度がworldのharvestの方が高そうだったので変更しました
    """
    hop_length = cfg.model.sampling_rate * cfg.model.frame_period // 1000  
    
    power = librosa.feature.rms(y=wav, frame_length=hop_length*2, hop_length=hop_length).squeeze()
    power = fill_nan(power)
    f0 = swipe(wav.astype("float64"), cfg.model.sampling_rate, hop_length, min=70.0, otype='f0').squeeze()

    f0, vuv = continuous_f0(f0)
    T = min(power.size, f0.size, feature.shape[0])
    
    feat_add = np.vstack((f0[:T], vuv[:T], power[:T])).T
    feat_add = np.log(np.maximum(feat_add, 1.0e-7))
    return feat_add, T


def calc_feat_add(wav, feature, cfg, use_spec=False):
    """
    音声のrms(root mean square)とclf0(continuous log f0)を計算
    multi task learningなどに使用できる
    use_specでrmsを音声波形から計算するかスペクトログラムから計算するかを選択
    一応librosaにはスペクトログラムから計算した方が精度がいいと書いていたけど,音声波形からでも十分そうだった
    """
    # rms
    if use_spec:
        spec = librosa.stft(
            y=wav,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
            window="hann",
        )
        spec_mag, spec_phase = librosa.magphase(spec)
        rms = librosa.feature.rms(S=spec_mag, frame_length=cfg.model.hop_length*4, hop_length=cfg.model.hop_length).squeeze()
    else:
        rms = librosa.feature.rms(y=wav, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
    rms = fill_nan(rms)
    rms = librosa.amplitude_to_db(rms, ref=np.max)

    # worldの推定手法(harvest)で連続対数基本周波数を計算
    mcep, clf0, vuv, cap, fbin, _ = wav2world(
        wave=wav, fs=cfg.model.sampling_rate, frame_period=cfg.model.frame_period, cfg=cfg,
    )

    T = min(rms.shape[0], clf0.shape[0], feature.shape[0])
    feat_add = np.vstack((clf0[:T], rms[:T])).T   # (T, C)
    return feat_add, T


def load_data_for_npz(video_path, audio_path, landmark_path, cfg):
    """
    lip : (C, H, W, T)
    feature, feat_add : (T, C)
    """
    lip, fps = load_mp4(str(video_path), cfg)   # lipはtensor
    wav, fs = librosa.load(str(audio_path), sr=cfg.model.sampling_rate, mono=None)
    wav = wav / np.max(np.abs(wav), axis=0)
    upsample = get_upsample(cfg)
    landmark = load_landmark(landmark_path)     # (T, 2, 68)
    
    # 音響特徴量への変換
    feature = calc_sp(wav, cfg)
    feat_add, T = calc_feat_add(wav, feature, cfg)
    feature = feature[:T]

    data_len = min(len(feature) // upsample * upsample,  lip.shape[-1] * upsample)
    feature = feature[:data_len]
    feat_add = feat_add[:data_len]
    lip = lip[..., :data_len // upsample]
    landmark = landmark[:data_len // upsample, ...]

    lip = lip.numpy()
    return wav, lip, feature, feat_add, upsample, data_len, landmark


def preprocess_movie(video_path, bbox_path, landmark_path, cfg, aligner):
    lip, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec")    # (T, W, H, C)

    if cfg.model.mov_preprocess_method == "align":
        lip = lip.permute(0, 2, 3, 1).numpy()   # (T, H, W, C)
        coords_list = get_landmark(landmark_path)

        lip_processed_list = []
        for i in range(lip.shape[0]):
            lip_processed = aligner.align(lip[i], coords_list[i])
            lip_processed_list.append(lip_processed)
        lip_processed = np.stack(lip_processed_list, axis=0)
        lip_processed = torch.from_numpy(lip_processed)     # (T, H, W, C)
        lip_processed = lip_processed.permute(0, 3, 1, 2)   # (T, C, H, W)

    elif cfg.model.mov_preprocess_method == "bbox_crop":
        lip = lip.permute(0, -1, 1, 2)  # (T, C, H, W)
        coords_mean, crop_size = get_crop_info(bbox_path, cfg.model.margin)
        if crop_size > lip.shape[-1]:
            crop_size = lip.shape[-1]
            crop_size = crop_size // 2 * 2

        lip_processed_list = []
        for i in range(lip.shape[0]):
            top = np.clip(coords_mean[i][1] - crop_size // 2, a_min=0, a_max=lip.shape[2] - crop_size)
            left = np.clip(coords_mean[i][0] - crop_size // 2, a_min=0, a_max=lip.shape[3] - crop_size)
            lip_processed = torchvision.transforms.functional.crop(
                lip[i, ...],
                top=top,
                left=left,
                height=crop_size,
                width=crop_size,
            )
            lip_processed_list.append(lip_processed)
            
        lip_processed = torch.stack(lip_processed_list, dim=0)      # (T, C, H, W)
        
    return lip_processed


def load_data(video_path, audio_path, bbox_path, landmark_path, text_path, cfg, aligner):
    # 動画から検出済みのbouding boxを利用して顔領域を切り取り
    lip = preprocess_movie(video_path, bbox_path, landmark_path, cfg, aligner)

    # グレースケール化 & リサイズ
    if cfg.model.gray:
        lip = torchvision.transforms.functional.rgb_to_grayscale(lip)   # (T, C, H, W)
    lip = torchvision.transforms.functional.resize(lip, [cfg.model.imsize, cfg.model.imsize])   # (T, C, W, H)

    wav, fs = librosa.load(str(audio_path), sr=cfg.model.sampling_rate, mono=None)
    wav = wav / np.max(np.abs(wav), axis=0)
    feature = wav2mel(wav, cfg, ref_max=False)  # (C, T)

    # 系列長の調整
    upsample = get_upsample(cfg)
    data_len = min(int(feature.shape[1] // upsample * upsample), int(lip.shape[0] * upsample))
    lip = lip[:data_len // upsample,  ...]
    feature = feature[:, :data_len]
    n_wav_sample_per_frame = cfg.model.sampling_rate * cfg.model.frame_period // 1000
    wav = wav[:int(n_wav_sample_per_frame * data_len)]
    wav_padded = np.zeros(int(n_wav_sample_per_frame * data_len))
    wav_padded[:wav.shape[0]] = wav
    wav = wav_padded

    df = pd.read_csv(str(text_path))
    text = df.pronounce.values[0]

    assert feature.shape[1] == int(lip.shape[0] * upsample)
    lip = lip.permute(1, 2, 3, 0).numpy()   # (C, H, W, T)
    feature = feature.T     # (T, C)
    return wav, lip, feature, data_len, text


def load_data_lrs2(video_path, bbox_path, landmark_path, cfg, aligner):
    """
    wav : (T,)
    lip : (C, H, W, T)
    feature : (T, C)
    """
    # 動画から検出済みのbouding boxを利用して顔領域を切り取り
    lip = preprocess_movie(video_path, bbox_path, landmark_path, cfg, aligner)

    # グレースケール化 & リサイズ
    if cfg.model.gray:
        lip = torchvision.transforms.functional.rgb_to_grayscale(lip)   # (T, C, H, W)
    lip = torchvision.transforms.functional.resize(lip, [cfg.model.imsize, cfg.model.imsize])   # (T, C, W, H)

    # mp4から音声を読み込み
    wav, fs = librosa.load(str(video_path), sr=cfg.model.sampling_rate, mono=None)
    wav = wav / np.max(np.abs(wav), axis=0)
    feature = wav2mel(wav, cfg, ref_max=False)  # (C, T)

    # 系列長の調整
    upsample = get_upsample(cfg)
    data_len = min(feature.shape[1] // 4 * 4, int(lip.shape[0] * upsample))
    lip = lip[:data_len // upsample,  ...]
    feature = feature[:, :data_len]
    n_wav_sample_per_frame = cfg.model.sampling_rate * cfg.model.frame_period // 1000
    wav = wav[:int(n_wav_sample_per_frame * data_len)]
    wav_padded = np.zeros(int(n_wav_sample_per_frame * data_len))
    wav_padded[:wav.shape[0]] = wav
    wav = wav_padded

    assert feature.shape[1] == int(lip.shape[0] * upsample)
    lip = lip.permute(1, 2, 3, 0).numpy()   # (C, H, W, T)
    feature = feature.T     # (T, C)
    return wav, lip, feature, data_len