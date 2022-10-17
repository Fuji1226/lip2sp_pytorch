"""
データの保存を行う処理
口唇動画,音響特徴量,合成音声などを保存します
"""
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torchvision
from data_process.feature import mel2wav, world2wav, wav2mel, wav2world, wav2spec
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
import numpy as np
import pyworld

from data_process.phoneme_encode import get_keys_from_value


def save_lip_video(cfg, save_path, lip, lip_mean, lip_std):
    """
    口唇動画と動的特徴量の保存
    動的特徴量は無理やり可視化してるので適切かは分かりません…
    lip : (C, H, W, T)
    lip_mean, lip_std : (C, H, W) or (C,)
    """
    # gray scale
    if lip.shape[0] == 3:
        lip_orig = lip[:1, ...]    
    # rgb
    elif lip.shape[0] == 5:
        lip_orig = lip[:3, ...]
    lip_delta = lip[-2, ...].unsqueeze(0)
    lip_deltadelta = lip[-1, ...].unsqueeze(0)
    
    # 標準化したので元のスケールに直す
    if lip_std.dim() == 1:
        lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
        lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
        lip_orig = torch.mul(lip_orig, lip_std)
        lip_orig = torch.add(lip_orig, lip_mean)
        lip_delta = torch.mul(lip_delta, lip_std)
        lip_delta = torch.add(lip_delta, lip_mean)
        lip_deltadelta = torch.mul(lip_deltadelta, lip_std)
        lip_deltadelta = torch.add(lip_deltadelta, lip_mean)
    elif lip_std.dim() == 3:
        lip_std = lip_std.unsqueeze(-1)     # (C, H, W, 1)
        lip_mean = lip_mean.unsqueeze(-1)   # (C, H, W, 1)
        lip_orig = torch.mul(lip_orig, lip_std)
        lip_orig = torch.add(lip_orig, lip_mean)
        lip_delta = torch.mul(lip_delta, torch.mean(lip_std, dim=(1, 2)).unsqueeze(1).unsqueeze(1))
        lip_delta = torch.add(lip_delta, torch.mean(lip_mean, dim=(1, 2)).unsqueeze(1).unsqueeze(1))
        lip_deltadelta = torch.mul(lip_deltadelta, torch.mean(lip_std, dim=(1, 2)).unsqueeze(1).unsqueeze(1))
        lip_deltadelta = torch.add(lip_deltadelta, torch.mean(lip_mean, dim=(1, 2)).unsqueeze(1).unsqueeze(1))
    
    lip_orig = lip_orig.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
    lip_delta = lip_delta.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
    lip_deltadelta = lip_deltadelta.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
    lip_orig = lip_orig.to('cpu')
    lip_delta = lip_delta.to('cpu')
    lip_deltadelta = lip_deltadelta.to('cpu')

    torchvision.io.write_video(
        filename=str(save_path / "lip.mp4"),
        video_array=lip_orig,
        fps=cfg.model.fps
    )
    torchvision.io.write_video(
        filename=str(save_path / "lip_d.mp4"),
        video_array=lip_delta,
        fps=cfg.model.fps
    )
    torchvision.io.write_video(
        filename=str(save_path / "lip_dd.mp4"),
        video_array=lip_deltadelta,
        fps=cfg.model.fps
    )


def save_wav(cfg, save_path, file_name, feature, feat_mean, feat_std):
    """
    音響特徴量から音声波形を生成し、wavファイルを保存
    sharpを使用するとちょっと合成音声が綺麗になります
    feature : (C, T)
    feat_mean, feat_std : (C,)
    """
    # world特徴量
    if cfg.model.feature_type == "world":
        feature = feature.to('cpu').numpy()
        feat_mean = feat_mean.unsqueeze(1).to('cpu').numpy()
        feat_std = feat_std.unsqueeze(1).to('cpu').numpy()
        
        # 標準化したので元のスケールに直す
        feature *= feat_std
        feature += feat_mean

        feature = feature.T
        mcep = feature[:, :26]
        clf0 = feature[:, 26]
        vuv = feature[:, 27]
        cap = feature[:, 28:]

        wav = world2wav(
            sp=mcep,
            clf0=clf0,
            vuv=vuv,
            cap=cap,
            fs=cfg.model.sampling_rate,
            fbin=513,
            frame_period=cfg.model.frame_period,
            mcep_postfilter=True,
            cfg=cfg,
        )
        # 正規化
        wav /= np.max(np.abs(wav))
        write(str(save_path / f"world_{file_name}.wav"), rate=cfg.model.sampling_rate, data=wav)

    # メルスペクトログラム
    if cfg.model.feature_type == "mspec":
        feature = feature.to('cpu').numpy()
        feat_mean = feat_mean.unsqueeze(1).to('cpu').numpy()
        feat_std = feat_std.unsqueeze(1).to('cpu').numpy()

        # 標準化したので元のスケールに直す
        feature *= feat_std
        feature += feat_mean

        wav = mel2wav(feature, cfg)

        # 正規化
        wav /= np.max(np.abs(wav))
        write(str(save_path / f"mspec_{file_name}.wav"), rate=cfg.model.sampling_rate, data=wav)

    return wav


def plot_wav(cfg, save_path, wav_input, wav_AbS, wav_gen):
    """
    音声波形のプロット
    """
    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    time = np.arange(0, wav_input.shape[0]) / cfg.model.sampling_rate

    ax = plt.subplot(3, 1, 1)
    ax.plot(time, wav_input)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.title("Input")
    plt.grid()

    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    ax.plot(time, wav_AbS)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.title("Analysis by Synthesis")
    plt.grid()

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    ax.plot(time, wav_gen)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.title("Synthesis")
    plt.grid()

    plt.tight_layout()
    plt.savefig(str(save_path / "waveform.png"))


def plot_mel(cfg, save_path, wav_input, wav_AbS, wav_gen, ref_max=True):
    """
    メルスペクトログラムのプロット
    """
    mel_input = wav2mel(wav_input, cfg, ref_max=ref_max)
    mel_AbS = wav2mel(wav_AbS, cfg, ref_max=ref_max)
    mel_gen = wav2mel(wav_gen, cfg, ref_max=ref_max)

    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    ax = plt.subplot(3, 1, 1)
    specshow(
        data=mel_input, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Input")
    
    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=mel_AbS, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Analysis by Synthesis")

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    specshow(
        data=mel_gen, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Synthesis")

    plt.tight_layout()
    plt.savefig(str(save_path / "melspectrogram.png"))


def plot_spec(cfg, save_path, wav_input, wav_AbS, wav_gen, ref_max=True):
    """
    スペクトログラムのプロット
    """
    spec_input = wav2spec(wav_input, cfg, ref_max=ref_max)
    spec_AbS = wav2spec(wav_AbS, cfg, ref_max=ref_max)
    spec_gen = wav2spec(wav_gen, cfg, ref_max=ref_max)

    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    ax = plt.subplot(3, 1, 1)
    specshow(
        data=spec_input, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Input")
    
    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=spec_AbS, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Analysis by Synthesis")

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    specshow(
        data=spec_gen, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Synthesis")

    plt.tight_layout()
    plt.savefig(str(save_path / "spectrogram.png"))


def plot_spec_envelope(cfg, save_path, spec_input, spec_AbS, spec_gen):
    """
    WORLDから算出されたスペクトル包絡のプロット
    """
    spec_input = librosa.power_to_db(spec_input, ref=np.max)
    spec_AbS = librosa.power_to_db(spec_AbS, ref=np.max)
    spec_gen = librosa.power_to_db(spec_gen, ref=np.max)

    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    ax = plt.subplot(3, 1, 1)
    specshow(
        data=spec_input, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Input")
    
    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=spec_AbS, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Analysis by Synthesis")

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    specshow(
        data=spec_gen, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Synthesis")

    plt.tight_layout()
    plt.savefig(str(save_path / "spectral_envelope.png"))


def plot_f0(cfg, save_path, f0_input, f0_AbS, f0_gen):
    """
    基本周波数のプロット
    """
    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    time = np.arange(0, f0_input.shape[0]) / 100

    ax = plt.subplot(3, 1, 1)
    ax.plot(time, f0_input)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[hz]")
    plt.title("Input")
    plt.grid()

    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    ax.plot(time, f0_AbS)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[hz]")
    plt.title("Analysis by Synthesis")
    plt.grid()

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    ax.plot(time, f0_gen)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[hz]")
    plt.title("Synthesis")
    plt.grid()

    plt.tight_layout()
    plt.savefig(str(save_path / "f0.png"))


def plot_f0_from_wav(cfg, save_path, wav_input, wav_AbS, wav_gen, f0_floor=None, f0_ceil=None):
    """
    音声波形からf0を計算し,その上でプロットする
    """
    wav_input = wav_input.astype('float64')
    wav_AbS = wav_AbS.astype('float64')
    wav_gen = wav_gen.astype('float64')

    f0_floor = pyworld.default_f0_floor if f0_floor is None else f0_floor
    f0_ceil = pyworld.default_f0_ceil if f0_ceil is None else f0_ceil

    f0_input, _ = pyworld.harvest(
        wav_input, 
        cfg.model.sampling_rate,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=cfg.model.frame_period,
    )
    f0_AbS, _ = pyworld.harvest(
        wav_AbS, 
        cfg.model.sampling_rate,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=cfg.model.frame_period,
    )
    f0_gen, _ = pyworld.harvest(
        wav_gen, 
        cfg.model.sampling_rate,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=cfg.model.frame_period,
    )
    
    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    time = np.arange(0, f0_input.shape[0]) / 100

    ax = plt.subplot(3, 1, 1)
    ax.plot(time, f0_input)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[hz]")
    plt.title("Input")
    plt.grid()

    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    ax.plot(time, f0_AbS)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[hz]")
    plt.title("Analysis by Synthesis")
    plt.grid()

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    ax.plot(time, f0_gen)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[hz]")
    plt.title("Synthesis")
    plt.grid()

    plt.tight_layout()
    plt.savefig(str(save_path / "f0.png"))


def plot_vuv(cfg, save_path, vuv_input, vuv_AbS, vuv_gen):
    """
    有声無声判定のプロット
    """
    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    time = np.arange(0, vuv_input.shape[0]) / 100

    ax = plt.subplot(3, 1, 1)
    ax.plot(time, vuv_input)
    plt.xlabel("Time[s]")
    plt.title("Input")
    plt.grid()

    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    ax.plot(time, vuv_AbS)
    plt.xlabel("Time[s]")
    plt.title("Analysis by Synthesis")
    plt.grid()

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    ax.plot(time, vuv_gen)
    plt.xlabel("Time[s]")
    plt.title("Synthesis")
    plt.grid()

    plt.tight_layout()
    plt.savefig(str(save_path / "vuv.png"))


def plot_ap(cfg, save_path, ap_input, ap_AbS, ap_gen):
    """
    非周期性指標のプロット
    """
    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    ax = plt.subplot(3, 1, 1)
    specshow(
        data=ap_input, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar()
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Input")
    
    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=ap_AbS, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar()
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Analysis by Synthesis")

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    specshow(
        data=ap_gen, 
        x_axis="time", 
        y_axis="linear", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar()
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("Synthesis")

    plt.tight_layout()
    plt.savefig(str(save_path / "aperiodicity.png"))


def plot_world(cfg, save_path, wav_input, wav_AbS, wav_gen):
    """
    WORLD特徴量のプロット
    """
    # 音声波形からWORLD特徴量を計算
    spec_input, f0_input, vuv_input, ap_input = wav2world(
        wave=wav_input,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        cfg=cfg,
        sp_type="spec",
        plot=True,
    )
    spec_AbS, f0_AbS, vuv_AbS, ap_AbS = wav2world(
        wave=wav_AbS,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        cfg=cfg,
        sp_type="spec",
        plot=True,
    )
    spec_gen, f0_gen, vuv_gen, ap_gen = wav2world(
        wave=wav_gen,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        cfg=cfg,
        sp_type="spec",
        plot=True,
    )

    # それぞれの特徴量をプロット
    plot_spec_envelope(cfg, save_path, spec_input.T, spec_AbS.T, spec_gen.T)
    plot_f0(cfg, save_path, f0_input, f0_AbS, f0_gen)
    plot_vuv(cfg, save_path, vuv_input, vuv_AbS, vuv_gen)
    plot_ap(cfg, save_path, ap_input.T, ap_AbS.T, ap_gen.T)


def save_data(cfg, save_path, wav, lip, feature, feat_add, output, lip_mean, lip_std, feat_mean, feat_std, enhanced_output=None):
    """
    出力データの保存
    """
    wav = wav.squeeze(0)
    lip = lip.squeeze(0)
    feature = feature.squeeze(0)
    feat_add = feat_add.squeeze(0)
    output = output.squeeze(0)

    wav = wav.to('cpu').numpy()

    # 口唇動画の保存
    save_lip_video(
        cfg=cfg,
        save_path=save_path,
        lip=lip,
        lip_mean=lip_mean,
        lip_std=lip_std
    )

    # 原音声のwavファイルを保存
    write(str(save_path / "input.wav"), rate=cfg.model.sampling_rate, data=wav)

    # 原音声の音響特徴量から求めた合成音声のwavファイルを保存
    wav_AbS = save_wav(
        cfg=cfg,
        save_path=save_path,
        file_name="AbS",
        feature=feature,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )
    # モデルで推定した音響特徴量から求めた合成音声のwavファイルを保存
    wav_gen = save_wav(
        cfg=cfg,
        save_path=save_path,
        file_name="generate",
        feature=output,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )

    # サンプル数を合わせるための微調整
    n_sample = min(wav.shape[0], wav_AbS.shape[0], wav_gen.shape[0])
    wav = wav[:n_sample]
    wav_AbS = wav_AbS[:n_sample]
    wav_gen = wav_gen[:n_sample]
    assert wav.shape[0] == wav_AbS.shape[0] == wav_gen.shape[0]

    # プロット
    plot_wav(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_mel(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_spec(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_f0_from_wav(cfg, save_path, wav, wav_AbS, wav_gen)

    
def save_data_lipreading(cfg, save_path, wav, lip, lip_mean, lip_std, phoneme_index_output, output, classes_index):
    """
    出力データの保存
    lip reading用
    """
    # wav = wav.squeeze(0)
    lip = lip.squeeze(0)
    phoneme_index_output = phoneme_index_output.squeeze(0)
    output = output.squeeze(0)

    # 口唇動画の保存
    save_lip_video(
        cfg=cfg,
        save_path=save_path,
        lip=lip,
        lip_mean=lip_mean,
        lip_std=lip_std
    )

    # 原音声のwavファイルを保存
    # write(str(save_path / "input.wav"), rate=cfg.model.sampling_rate, data=wav)

    # 音素を数値列から元の音素ラベルに戻す
    phoneme_answer = []
    for i in phoneme_index_output:
        phoneme_answer.append(get_keys_from_value(classes_index, i))
    phoneme_answer = " ".join(phoneme_answer)
    
    phoneme_predict = []
    for i in output:
        phoneme_predict.append(get_keys_from_value(classes_index, i))
    phoneme_predict = " ".join(phoneme_predict)
    
    with open(str(save_path / "phoneme.txt"), "a") as f:
        f.write("answer\n")
        f.write(f"{phoneme_answer}\n")
        f.write("\npredict\n")
        f.write(f"{phoneme_predict}\n")
