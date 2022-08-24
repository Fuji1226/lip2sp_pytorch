"""
mel2wav/feature.py

wav2world、world2wavを追加

メルスペクトログラムから動的特徴量を求めるdelta_featureを追加

wave2mel, mel2wavをlibrosaに変更
"""

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
from zmq import device


OVERLAP = 4
EPS = 1.0e-8


def get_stft_params(fs, frame_period):
    nshift = fs * frame_period // 1000
    nperseg = nshift * OVERLAP
    noverlap = nperseg - nshift
    assert signal.check_COLA("hann", nperseg, noverlap)

    return nshift, nperseg, noverlap


def stft(x, fs, frame_period):
    _, nperseg, noverlap = get_stft_params(fs, frame_period)
    _, _, Zxx = signal.stft(x, fs=fs, window='hann',
                            nperseg=nperseg, noverlap=noverlap)
    return Zxx


def istft(Zxx, fs, frame_period):
    _, nperseg, noverlap = get_stft_params(fs, frame_period)
    _, x = signal.istft(Zxx, fs=fs, window='hann',
                        nperseg=nperseg, noverlap=noverlap)
    return x


def griffin_lim(H, fs, frame_period, n_iter=100, initial_phase=None, return_waveform=True):
    if initial_phase is None:
        initial_phase = (np.random.rand(
            *H.shape).astype(H.dtype) * 2 - 1) * np.pi
    assert H.shape == initial_phase.shape

    Zxx = H * np.exp(1j * initial_phase)
    for _ in range(n_iter):
        x = istft(Zxx, fs, frame_period)
        Zxx = stft(x, fs, frame_period)
        Zxx = H * Zxx / np.maximum(np.abs(Zxx), 1e-16)

    if return_waveform:
        return istft(Zxx, fs, frame_period)
    else:
        return Zxx


def fast_griffin_lim(H, fs, frame_period, alpha=0.99, n_iter=100, initial_phase=None):
    raise NotImplementedError
    if initial_phase is None:
        initial_phase = (np.random.rand(
            *H.shape).astype(H.dtype) * 2 - 1) * np.pi
    assert H.shape == initial_phase.shape

    c = H * np.exp(1j * initial_phase)
    x = istft(c, fs, frame_period)
    t_prev = stft(x, fs, frame_period)
    for _ in range(n_iter):
        x = istft(c, fs, frame_period)
        t = stft(x, fs, frame_period)
        c = t + alpha * (t - t_prev)
        c = H * c / np.maximum(np.abs(c), 1e-16)
        t_prev = t

    return istft(c, fs, frame_period)


def get_melfb(fs, frame_period, n_mels=80, fmin=70, fmax=None):
    _, nperseg, _ = get_stft_params(fs, frame_period)
    if n_mels is None:
        fb = np.eye(nperseg // 2 + 1)
    else:
        if fmax is None:
            fmax = float(fs) / 2 * 0.95

        fb = filters.mel(fs, nperseg, n_mels=n_mels,
                         fmin=fmin, fmax=fmax, htk=True, norm="slaney")

    return fb


def spec2mel(H, mel_fb=None, fs=None, frame_period=None, n_mels=80, fmin=70, fmax=None):
    mel_fb = get_melfb(fs, frame_period, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return mel_fb @ H


def wave2mel(wave, fs, frame_period, n_mels=80, fmin=70, fmax=None, return_linear=False):
    """
    wave : (T,)

    return
    ret : (C, T)
    """
    hop_length, win_length, _ = get_stft_params(fs, frame_period)
    y = librosa.feature.melspectrogram(
        y=wave,
        sr=fs,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    y = librosa.power_to_db(y, ref=np.max)
    ret = y

    return ret


def log10(x, eps=EPS):
    return np.log10(np.maximum(x, eps))


def mel2wave(
        Hmel, fs, frame_period,
        fmin=70, fmax=None, n_iter=50,
        sharpen=np.sqrt(1.4), eps=EPS):

    hop_length, win_length, _ = get_stft_params(fs, frame_period)
    spec = librosa.db_to_power(Hmel)
    wave = librosa.feature.inverse.mel_to_audio(
        spec,
        sr=fs,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=n_iter
    )
    return wave


def linear_interp_1d(arr, factor, axis=-1):
    assert factor >= 1
    if factor == 1:
        return arr

    idx_x = np.arange(arr.shape[axis])
    idx_x = idx_x / idx_x.max()

    idx_y = np.arange(arr.shape[axis] * factor)
    idx_y = idx_y / idx_y.max()

    return interp1d(idx_x, arr, kind='nearest', axis=axis)(idx_y)


def melspec2linear(Hmel, time_factor, mel_fb=None, fs=None, frame_period=None, fmin=70, fmax=None):
    if time_factor < 1:
        raise ValueError
    elif time_factor > 1:
        Hmel = linear_interp_1d(Hmel, time_factor, axis=-1)

    if mel_fb is None:
        n_mels = Hmel.shape[0]
        mel_fb = get_melfb(fs, frame_period, n_mels=n_mels,
                           fmin=fmin, fmax=fmax)
    return nnls(mel_fb, Hmel)


def scale(x, in_scale, out_scale):
    ret = (x - in_scale[0]) / (in_scale[1] - in_scale[0])
    ret = ret * (out_scale[1] - out_scale[0]) + out_scale[0]
    return ret


def modspec_smoothing(array, fs, cut_off=30, axis=0, fbin=11):
    if cut_off >= fs / 2:
        return array
    h = signal.firwin(fbin, cut_off, nyq=fs // 2)
    return signal.filtfilt(h, 1, array, axis)


def wav2world(
        wave, fs,
        mcep_order=26, f0_smoothing=0,
        ap_smoothing=0, sp_smoothing=0,
        frame_period=None, f0_floor=None, f0_ceil=None,
        f0_mode="harvest", sp_type="mcep", comp_mode='melfb', plot=False):
    # setup default values
    wave = wave.astype('float64')
    assert comp_mode == 'default' or 'melfb'

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
    
    if comp_mode == 'default':
        cap = pyworld.code_aperiodicity(ap, fs)
    elif comp_mode == 'melfb':
        # メルフィルタバンクを適用することでn_melsまで帯域圧縮
        melfb = librosa.filters.mel(sr=fs, n_fft=1024, n_mels=4, fmin=0, fmax=7600)
        cap = np.matmul(melfb, ap.T).T  # (T, C)
    
    # coding sp
    if sp_type == "spec":
        sp = sp
    elif sp_type == "mcep":
        alpha = pysptk.util.mcepalpha(fs)
        sp = pysptk.mcep(sp, order=mcep_order-1, alpha=alpha, itype=4)
    elif sp_type == "mfcc":
        sp = pyworld.code_spectral_envelope(sp, fs, mcep_order)
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
        sp, clf0, vuv, cap, fs, fbin,
        frame_period=None, mcep_postfilter=False,
        sp_type="mcep", vuv_thr=0.5, comp_mode='melfb'):
    """
    input 
    all feature : (T, C)
    """
    assert comp_mode == 'melfb' or 'default'

    # setup
    frame_period = pyworld.default_frame_period \
        if frame_period is None else frame_period

    clf0 = np.ascontiguousarray(clf0.astype('float64'))
    vuv = np.ascontiguousarray(vuv > vuv_thr).astype('int')
    cap = np.ascontiguousarray(cap.astype('float64'))
    sp = np.ascontiguousarray(sp.astype('float64'))
    fft_len = fbin * 2 - 2

    # clf0 2 f0
    f0 = np.squeeze(np.exp(clf0)) * np.squeeze(vuv)

    # cap 2 ap
    if comp_mode == 'default':
        cap = np.minimum(cap, 0.0)
        if cap.ndim != 2:
            cap = np.expand_dims(cap, 1)
        ap = pyworld.decode_aperiodicity(cap, fs, fft_len)
        ap -= ap.min()
        ap /= ap.max()
    elif comp_mode == 'melfb':
        melfb = librosa.filters.mel(sr=fs, n_fft=1024, n_mels=4, fmin=0, fmax=7600)
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


def world2wav_direct(feature, feat_mean, feat_std, cfg):
    """
    train_d.pyで使用
    tensorで渡してこの中で変換
    tensorでwavを返す
    """
    batch, c, t = feature.shape
    feature = feature.to('cpu').detach().numpy().copy()
    feat_mean = feat_mean.unsqueeze(1).to('cpu').detach().numpy().copy()
    feat_std = feat_std.unsqueeze(1).to('cpu').detach().numpy().copy()
    
    # 正規化したので元のスケールに直す
    feature *= feat_std
    feature += feat_mean

    # wav_save = np.zeros((batch, cfg.model))
    wav_save = []
    breakpoint()
    for i in range(batch):
        f = feature[i]
        f = f.T
        mcep = f[:, :-3]
        clf0 = f[:, -3]
        vuv = f[:, -2]
        cap = f[:, -1]
        wav = world2wav(
            sp=mcep,
            clf0=clf0,
            vuv=vuv,
            cap=cap,
            fs=cfg.model.sampling_rate,
            fbin=513,
            frame_period=cfg.model.frame_period,
            mcep_postfilter=True,
        )
        # wav_save[i] = wav
        wav_save.append(wav)
    wav = np.array(wav_save)
    breakpoint()
    wav = torch.from_numpy(wav_save)
    return wav


##########################################
# add delta_feature
##########################################
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


##########################################
# add blur_pooling
##########################################
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




def main():
    batch = 1
    channels = 4
    frames = 3
    x = torch.rand(batch, channels, frames)
    print(x)
    order = 2
    out = delta_feature(x, order)
    print(out)

    x = torch.rand(batch, channels, frames)
    x = x.unsqueeze(1)
    blur_pooling2D(x)
    return


if __name__ == "__main__":
    main()