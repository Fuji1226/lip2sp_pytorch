import numpy as np
from librosa import filters
from librosa.util import nnls
from scipy import signal
from scipy.interpolate import interp1d

# add
import pyworld
from pyreaper import reaper
import pysptk
from nnmnkwii.postfilters import merlin_post_filter
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F


OVERLAP = 4
EPS = 1.0e-6


def log10(x, eps=EPS):
    """
    常用対数をとる
    epsでクリッピング
    """
    return np.log10(np.maximum(x, eps))


def wav2mel(wav, cfg, ref_max=False):
    """
    音声波形をメルスペクトログラムに変換
    wav : (T,)
    mel_spec : (C, T)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.model.sampling_rate,
        n_fft=cfg.model.n_fft,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        window="hann",
        n_mels=cfg.model.n_mel_channels,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
    )
    if ref_max:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    else:
        mel_spec =  log10(mel_spec)
    return mel_spec


def wav2spec(wav, cfg, ref_max=False):
    """
    音声波形を対数パワースペクトログラムに変換
    wav : (T,)
    spec : (C, T)
    """
    spec = librosa.stft(
        y=wav,
        n_fft=cfg.model.n_fft,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        window="hann",
    )
    spec = np.abs(spec) ** 2

    if ref_max:
        spec = librosa.power_to_db(spec, ref=np.max)
    else:
        spec = log10(spec)
    return spec
    

def calc_spec_and_mel(wav, cfg):
    """
    spec : 振幅スペクトログラム
    mel_spec : 対数メルスペクトログラム
    wav : (T,)
    spec, mel_spec : (C, T)
    """
    spec = librosa.stft(
        y=wav,
        n_fft=cfg.model.n_fft,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        window="hann",
    )

    # 振幅スペクトログラム
    spec = np.abs(spec)
    
    # パワー
    spec_power = spec ** 2

    # メルスペクトログラム
    mel_spec = librosa.feature.melspectrogram(
        S=spec_power,
        sr=cfg.model.sampling_rate,
        n_fft=cfg.model.n_fft,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        window="hann",
        n_mels=cfg.model.n_mel_channels,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
    )
    
    # 対数
    mel_spec = log10(mel_spec)
    return spec, mel_spec


def spec2mel(spec, cfg):
    """
    振幅スペクトログラムを対数メルスペクトログラムに変換
    """
    spec_power = spec ** 2
    mel_spec = librosa.feature.melspectrogram(
        S=spec_power,
        sr=cfg.model.sampling_rate,
        n_fft=cfg.model.n_fft,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        window="hann",
        n_mels=cfg.model.n_mel_channels,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
    )
    mel_spec = log10(mel_spec)
    return mel_spec


def mel2wav(mel, cfg):
    """
    対数メルスペクトログラムからgriffin limによる音声合成
    """
    # 振幅スペクトログラムへの変換
    mel = 10 ** mel
    mel = np.where(mel > EPS, mel, 0)
    spec = librosa.feature.inverse.mel_to_stft(
        M=mel,
        sr=cfg.model.sampling_rate,
        n_fft=cfg.model.n_fft,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
    )

    # ちょっと音声が強調される。田口さんからの継承。
    if cfg.model.sharp:
        spec **= np.sqrt(1.4)

    wav = librosa.griffinlim(
        S=spec,
        n_iter=100,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        n_fft=cfg.model.n_fft,
        window="hann",
    )
    return wav


def spec2wav(spec, cfg):
    spec = 10 ** spec
    spec = np.where(spec > EPS, spec, 0)
    wav = librosa.griffinlim(
        S=spec,
        n_iter=100,
        hop_length=cfg.model.hop_length,
        win_length=cfg.model.win_length,
        n_fft=cfg.model.n_fft,
        window="hann",
    )
    return wav


def modspec_smoothing(array, fs, cut_off=30, axis=0, fbin=11):
    if cut_off >= fs / 2:
        return array
    h = signal.firwin(fbin, cut_off, nyq=fs // 2)
    return signal.filtfilt(h, 1, array, axis)


def wav2world(
        wave, fs, cfg,
        mcep_order=25, f0_smoothing=0,
        ap_smoothing=0, sp_smoothing=0,
        frame_period=None, f0_floor=None, f0_ceil=None,
        f0_mode="harvest", sp_type="mcep", plot=False):
    """
    音声波形からWORLD特徴量を計算
    default_frame_period = 5.0
    default_f0_floor = 71.0
    default_f0_ceil = 800.0
    """
    # setup default values
    wave = wave.astype('float64')

    frame_period = pyworld.default_frame_period \
        if frame_period is None else frame_period
    f0_floor = pyworld.default_f0_floor if f0_floor is None else f0_floor       
    f0_ceil = pyworld.default_f0_ceil if f0_ceil is None else f0_ceil
    
    # f0
    if f0_mode == "harvest":
        f0, t = pyworld.harvest(
            wave, fs,
            f0_floor=f0_floor, f0_ceil=f0_ceil,
            frame_period=frame_period)
        threshold = 0.85
    
    elif f0_mode == "reaper":
        _, _, t, f0, _ = reaper(
            (wave * (2**15 - 1)).astype("int16"),
            fs, frame_period=frame_period / 1000,
            do_hilbert_transform=True)
        t, f0 = t.astype('float64'), f0.astype('float64')
        threshold = 0.1

    elif f0_mode == "dio":
        _f0, t = pyworld.dio(wave, fs, frame_period=frame_period)
        f0 = pyworld.stonemask(wave, _f0, t, fs)
        threshold = 0.0

    else:
        raise ValueError
    
    # world
    sp = pyworld.cheaptrick(wave, f0, t, fs)   
    ap = pyworld.d4c(wave, f0, t, fs, threshold=threshold)
    fbin = sp.shape[1]
    
    # extract vuv from ap
    vuv_flag = (ap[:, 0] < 0.5) * (f0 > 1.0)
    vuv = vuv_flag.astype('int')
    
    # continuous log f0
    clf0 = np.zeros_like(f0)
    if vuv_flag.any():
        if not vuv_flag[0]:
            f0[0] = f0[vuv_flag][0]
            vuv_flag[0] = True
        if not vuv_flag[-1]:
            f0[-1] = f0[vuv_flag][-1]
            vuv_flag[-1] = True

        idx = np.arange(len(f0))
        clf0[idx[vuv_flag]] = np.log(
            np.clip(f0[idx[vuv_flag]], f0_floor / 2, f0_ceil * 2))
        clf0[idx[~vuv_flag]] = interp1d(
            idx[vuv_flag], clf0[idx[vuv_flag]]
        )(idx[~vuv_flag])

    else:
        clf0 = np.ones_like(f0) * f0_floor
    
    if cfg.model.comp_mode == 'default':
        # 中心周波数3,6,9,12,15kHzで、それぞれ6kHzの帯域を分析対象として非周期性指標を帯域ごとに圧縮する
        # 公式の処理
        # 本研究ではサンプリング周波数16kHzなので8kHzまでが分析の対象であり、この場合中心周波数3kHzで0-6kHzのみが分析する帯域となるので1次元に削減される
        cap = pyworld.code_aperiodicity(ap, fs)
    elif cfg.model.comp_mode == 'melfb':
        # メルフィルタバンクを利用した非周期性指標の圧縮
        # 江崎さんが使用されていた
        melfb = librosa.filters.mel(
            sr=fs, n_fft=1024, n_mels=cfg.model.n_mel_fb, fmin=cfg.model.f_min, fmax=cfg.model.f_max
        )
        cap = np.matmul(melfb, ap.T).T  # (T, C)
    
    # coding sp
    if sp_type == "spec":
        sp = sp
    elif sp_type == "mcep":
        alpha = pysptk.util.mcepalpha(fs)
        sp = pysptk.mcep(sp, order=cfg.model.mcep_order-1, alpha=alpha, itype=4)
    elif sp_type == "mfcc":
        sp = pyworld.code_spectral_envelope(sp, fs, cfg.model.mcep_order)
    else:
        raise ValueError(sp_type)

    # apply mod spec smpoothing
    if sp_smoothing > 0:
        sp = modspec_smoothing(
            sp, 1000 / frame_period, cut_off=sp
        )
    if ap_smoothing > 0:
        cap = modspec_smoothing(cap, 1000 / frame_period, cut_off=ap_smoothing)
    if f0_smoothing > 0:
        clf0 = modspec_smoothing(
            clf0, 1000 / frame_period, cut_off=f0_smoothing)
    
    if plot:
        return sp.astype(np.float32), f0.astype(np.float32), vuv.astype(np.float32), ap.astype(np.float32)
    else:
        return sp, clf0, vuv, cap, fbin, t


def world2wav(
        sp, clf0, vuv, cap, fs, fbin, cfg,
        frame_period=None, mcep_postfilter=False,
        sp_type="mcep", vuv_thr=0.5):
    """
    input 
    all feature : (T, C)
    """
    # setup
    frame_period = pyworld.default_frame_period \
        if frame_period is None else frame_period

    clf0 = np.ascontiguousarray(clf0.astype('float64'))
    vuv = np.ascontiguousarray(vuv > vuv_thr).astype('int')     # 閾値を境に0,1の2値に分ける
    cap = np.ascontiguousarray(cap.astype('float64'))
    sp = np.ascontiguousarray(sp.astype('float64'))
    fft_len = fbin * 2 - 2

    # clf0 2 f0
    f0 = np.squeeze(np.exp(clf0)) * np.squeeze(vuv)

    # cap 2 ap
    if cfg.model.comp_mode == 'default':
        cap = np.minimum(cap, 0.0)
        if cap.ndim != 2:
            cap = np.expand_dims(cap, 1)
        ap = pyworld.decode_aperiodicity(cap, fs, fft_len)
        ap -= ap.min()
        ap /= ap.max()
    elif cfg.model.comp_mode == 'melfb':
        melfb = librosa.filters.mel(
            sr=fs, n_fft=1024, n_mels=cfg.model.n_mel_fb, fmin=cfg.model.f_min, fmax=cfg.model.f_max
        )
        melfb = np.ascontiguousarray(melfb.astype('float64'))
        ap = librosa.util.nnls(melfb, cap.T).T  # (T, C)
        ap = np.ascontiguousarray(ap.astype('float64'))

    # mcep 2 sp
    if sp_type == "spec":
        sp = sp
    elif sp_type == "mcep":
        alpha = pysptk.util.mcepalpha(fs)
        if mcep_postfilter:
            mcep = merlin_post_filter(sp, alpha)
        sp = pysptk.mgc2sp(mcep, alpha=alpha, fftlen=fft_len)
        sp = np.abs(np.exp(sp)) ** 2
    elif sp_type == "mfcc":
        sp = pyworld.decode_spectral_envelope(sp, fs, fft_len)
    else:
        raise ValueError(sp_type)

    wave = pyworld.synthesize(f0, sp, ap, fs, frame_period=frame_period)

    scale = np.abs(wave).max()
    if scale > 0.99:
        wave = wave / scale * 0.99

    return wave


def delta_feature(x, order=2, static=True, delta=True, deltadelta=True):
    """
    lip2sp/links/modules.pyにて動的特徴量の計算に使用されている、
    lip2sp/submodules/mychainer/utils/function.pyの関数delta_featureの実装（pytorchでの再現）

    x : (B, C, T)

    return
    out : (B, 3 * C, T) 

    staticに加えて、delta, deltadelta特徴量が増える分、チャンネル数が3倍になります
    """
    x = x.unsqueeze(1)  # (B, 1, C, T)

    # 動的特徴量を求めるためのフィルタを設定
    ws = []
    if order == 2:
        if static:
            ws.append(torch.tensor((0, 1, 0)))
        if delta:
            ws.append(torch.tensor((-1, 0, 1)) / 2)
        if deltadelta:
            ws.append(torch.tensor((1.0, -2.0, 1.0)))
        pad = 1

    elif order == 4:
        if static:
            ws.append(torch.tensor((0, 0, 1, 0, 0)))
        if delta:
            ws.append(torch.tensor((1, -8, 0, 8, -1)) / 12)
        if deltadelta:
            ws.append(torch.tensor((-1, 16, -30, 16, -1)) / 12)
        pad = 2

    else:
        raise ValueError(f"order: {order}")

    W = torch.stack(ws, dim=0).to(device=x.device)  # (3, 3)
    W = W.unsqueeze(1).unsqueeze(1)     # (3, 1, 1, 3) : (out_channels, in_channels, kernel_size_C, kernel_size_T)

    padding = nn.ReflectionPad2d(pad)

    # チャンネル方向のパディングはいらないので、取り除いてます
    x = padding(x)[:, :, pad:-1, :]     # (B, 1, C, T + 2)
    
    # 設定したフィルタでxに対して2次元畳み込みを行い、静的特徴量からdelta, deltadelta特徴量を計算
    out = F.conv2d(x, W)    # (B, 3, C, T)

    B, T = out.shape[0], out.shape[-1]
    out = out.view(B, -1, T)    # (B, 3 * C, T)

    return out


def blur_pooling2D(x, device, ksize=3, stride=1):
    """
    input
    x : (B, C, T)

    return
    out :(B, C, T)
    """

    x = x.unsqueeze(1)

    # pad_sizes = [(int(1.*(ksize-1)/2), int(np.ceil(1.*(ksize-1)/2))),
    #              (int(1.*(ksize-1)/2), int(np.ceil(1.*(ksize-1)/2)))]
    # pad_width = [(0, 0), (0, 0)] + pad_sizes

    # if ksize == 1:
    #     a = torch.tensor([1., ])
    # elif ksize == 2:
    #     a = torch.tensor([1., 1.])
    if ksize == 3:
        a = torch.tensor([1., 2., 1.])
    # elif ksize == 4:
    #     a = torch.tensor([1., 3., 3., 1.])
    # elif ksize == 5:
    #     a = torch.tensor([1., 4., 6., 4., 1.])
    # elif ksize == 6:
    #     a = torch.tensor([1., 5., 10., 10., 5., 1.])
    # elif ksize == 7:
    #     a = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise NotImplementedError(f"ksize: {ksize}")

    filt = a.unsqueeze(1) * a.unsqueeze(0)
    filt /= filt.sum()
    filt = filt.unsqueeze(0).unsqueeze(0)
    filt = filt.repeat(x.shape[1], 1, 1, 1).to(device)
    padding = nn.ReflectionPad2d(1)
    x = padding(x)

    out = F.conv2d(x, filt, stride=(stride, stride), groups=x.shape[1])
    out = out.squeeze(1)
    return out