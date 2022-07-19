"""
データのロード、前処理
make_npz.pyを実行するとここの処理が行われます
"""

import os
import sys
import glob

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import wave
from pathlib import Path
import cv2
import numpy as np
import torch
import librosa
from zipfile import BadZipFile
from scipy.interpolate import interp1d
from pysptk import swipe
import torchvision

from utils import get_sp_name, get_upsample
from data_process.feature import wave2mel, wav2world


def cals_sp(wave, fs, frame_period, feature_type, cfg, path=None, nmels=None, f_min=None, f_max=None):
    if feature_type == "mspec":
        # 対数メルスペクトログラム
        y = wave2mel(wave, fs, frame_period,
                        n_mels=nmels, fmin=f_min, fmax=f_max).T    # (T, C)

    elif feature_type == "world":
        # mcep, clf0, vuv, cap, _, fbin, _
        # 元々こうなっていましたが、wav2worldの引数が6つでバグったので合わせました。
        # 先輩のが謎です
        mcep, clf0, vuv, cap, fbin, _ = wav2world(
            wave, fs, frame_period=frame_period, comp_mode=cfg.model.comp_mode)
        y = np.hstack([mcep, clf0.reshape(-1, 1),
                        vuv.reshape(-1, 1), cap])
    else:
        raise ValueError(feature_type)

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
    可視化が楽だったので、torchvisionを利用して読み込むように変更
    リサイズで画像のピクセル数を変更
    """
    movie = cv2.VideoCapture(str(path))
    fps = int(movie.get(cv2.CAP_PROP_FPS))
    movie.release()

    ######################################################################
    lip, _, _ = torchvision.io.read_video(str(path))    # lip : (T, W, H, C)
    # torchvision.io.write_video(
    #     filename="/users/minami/dataset"+"/non_resize.mp4",
    #     video_array=lip,
    #     fps=fps
    # )
    resizer = torchvision.transforms.Resize((48, 48))
    lip_resize = resizer(lip.permute(0, -1, 1, 2))  # (T, C, W, H)
    # torchvision.io.write_video(
    #     filename="/users/minami/dataset"+"/resize.mp4",
    #     video_array=lip_resize.permute(0, 2, 3, 1),
    #     fps=fps
    # )

    lip_resize = lip_resize.permute(1, 2, 3, 0)  # (C, W, H, T)
    ######################################################################

    # mov = vread(str(path), as_grey=gray, outputdict={"-s": "48x48"})
    # mov = np.asarray(mov).swapaxes(0, -1)   # (C, W, H, T)
    
    return lip_resize, fps


def load_data(train, data_path, cfg, gray, frame_period, feature_type, nmels, f_min, f_max, mode=None, delta=True, return_wave=False):
    try:
        npz_key = np.load(f'{cfg.train.test_pre_loaded_path}/{cfg.model.name}/{data_path.stem}.npz')
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        # ファイルがある場合、それを読み込む
        # if train:
        #     npz_key = np.load(f'{cfg.train.train_pre_loaded_path}/{cfg.model.name}/{data_path.stem}.npz')
        #     lip = torch.from_numpy(npz_key['lip'])
        #     feature = torch.from_numpy(npz_key['feature'])
        #     feat_add = torch.from_numpy(npz_key['feat_add'])
        #     upsample = torch.from_numpy(npz_key['upsample'])
        #     data_len = torch.from_numpy(npz_key['data_len'])
        # else:
        #     npz_key = np.load(f'{cfg.train.test_pre_loaded_path}/{cfg.model.name}/{data_path.stem}.npz')
        #     lip = torch.from_numpy(npz_key['lip'])
        #     feature = torch.from_numpy(npz_key['feature'])
        #     feat_add = torch.from_numpy(npz_key['feat_add'])
        #     upsample = torch.from_numpy(npz_key['upsample'])
        #     data_len = torch.from_numpy(npz_key['data_len'])

        ret = (lip, feature, feat_add, upsample)
    except:
        lip, fps = load_mp4(str(data_path), gray)   # lipはtensor
        sppath = Path(data_path)
        sppath = sppath.parent / (sppath.stem + ".wav")
        wave, fs = librosa.load(str(sppath), sr=None, mono=None)
        wave = wave[:int(lip.shape[-1]/fps*1.2*fs)]
        upsample = get_upsample(fps, fs, frame_period)
        
        # 音響特徴量への変換
        feature = cals_sp(
            wave, fs, frame_period, feature_type,
            path=data_path, nmels=nmels, f_min=f_min, f_max=f_max)
        hop_length = fs * frame_period // 1000  # 160
        
        power = librosa.feature.rms(wave, frame_length=hop_length*2,
                    hop_length=hop_length).squeeze()
        power = fill_nan(power)
        f0 = swipe(wave.astype("float64"), fs, hop_length,
                min=70.0, otype='f0').squeeze()

        f0, vuv = continuous_f0(f0)
        T = min(power.size, f0.size, feature.shape[0])
        
        feat_add = np.vstack((f0[:T], vuv[:T], power[:T])).T
        feat_add = np.log(np.maximum(feat_add, 1.0e-7))
        feature = feature[:T]

        data_len = min(len(feature) // upsample * upsample,  lip.shape[-1] * upsample)
        feature = feature[:data_len]
        feat_add = feat_add[:data_len]
        lip = lip[..., :data_len // upsample]

        # 読み込んだデータをnumpyファイルで保存
        lip = lip.to('cpu').detach().numpy().copy()

        # if train:
        #     np.savez(
        #         f'{cfg.train.train_pre_loaded_path}/{cfg.model.name}/{Path(data_path).stem}',
        #         lip=lip,
        #         feature=feature,
        #         feat_add=feat_add,
        #         upsample=upsample,
        #         data_len=data_len
        #     )
        # else:
        #     np.savez(
        #         f'{cfg.train.test_pre_loaded_path}/{cfg.model.name}/{Path(data_path).stem}',
        #         lip=lip,
        #         feature=feature,
        #         feat_add=feat_add,
        #         upsample=upsample,
        #         data_len=data_len
        #     )

        np.savez(
            f'{cfg.train.train_pre_loaded_path}/{cfg.model.name}/{Path(data_path).stem}',
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            upsample=upsample,
            data_len=data_len
        )

        # numpy -> tensor
        lip = torch.from_numpy(lip)
        feature = torch.from_numpy(feature)
        feat_add = torch.from_numpy(feat_add)
        data_len = torch.tensor(data_len)
        upsample = torch.tensor(upsample)

        ret = (
            lip, feature, feat_add, upsample
        )

    if return_wave:
        return ret, data_len, wave
    else:
        return ret, data_len


def load_data_for_npz(video_path, audio_path, cfg, gray, frame_period, feature_type, nmels, f_min, f_max, comp_mode=None, delta=True, return_wave=False):

    lip, fps = load_mp4(str(video_path), gray)   # lipはtensor
    wave, fs = librosa.load(str(audio_path), sr=None, mono=None)
    wave = wave[:int(lip.shape[-1]/fps*1.2*fs)]
    upsample = get_upsample(fps, fs, frame_period)
    
    # 音響特徴量への変換
    feature = cals_sp(
        wave, fs, frame_period, feature_type, cfg,
        path=audio_path, nmels=nmels, f_min=f_min, f_max=f_max)
    hop_length = fs * frame_period // 1000  
    
    power = librosa.feature.rms(wave, frame_length=hop_length*2,
                hop_length=hop_length).squeeze()
    power = fill_nan(power)
    f0 = swipe(wave.astype("float64"), fs, hop_length,
            min=70.0, otype='f0').squeeze()

    f0, vuv = continuous_f0(f0)
    T = min(power.size, f0.size, feature.shape[0])
    
    feat_add = np.vstack((f0[:T], vuv[:T], power[:T])).T
    feat_add = np.log(np.maximum(feat_add, 1.0e-7))
    feature = feature[:T]

    data_len = min(len(feature) // upsample * upsample,  lip.shape[-1] * upsample)
    feature = feature[:data_len]
    feat_add = feat_add[:data_len]
    lip = lip[..., :data_len // upsample]

    lip = lip.to('cpu').detach().numpy().copy()

    ret = (lip, feature, feat_add, upsample)

    return ret, data_len
