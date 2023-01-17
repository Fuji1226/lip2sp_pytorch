"""
データの保存を行う処理
口唇動画,音響特徴量,合成音声などを保存します
"""
import sys
from pathlib import Path
import os
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torchvision
from data_process.feature import mel2wav, world2wav, wav2mel, wav2world, wav2spec
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
from librosa.display import specshow
import numpy as np
import seaborn as sns
import pyworld
import cv2
from jiwer import wer

from data_process.phoneme_encode import get_keys_from_value


def save_lip_video(cfg, save_path, lip, lip_mean, lip_std):
    """
    口唇動画と動的特徴量の保存
    動的特徴量は無理やり可視化してるので適切かは分かりません…
    lip : (C, H, W, T)
    lip_mean, lip_std : (C, H, W) or (C,)
    """
    if cfg.model.delta:
        # gray scale
        if lip.shape[0] == 3:
            lip_orig = lip[:1, ...]    
            lip_delta = lip[-2, ...].unsqueeze(0)
            lip_deltadelta = lip[-1, ...].unsqueeze(0)
        # rgb
        elif lip.shape[0] == 9:
            lip_orig = lip[:3, ...]
            lip_delta = lip[3:6, ...]
            lip_deltadelta = lip[6:, ...]
    
        # 標準化したので元のスケールに直す
        if lip_std.dim() == 1:
            lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
            lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
            lip_orig = (lip_orig * lip_std) + lip_mean
            lip_delta = (lip_delta * lip_std) + lip_mean
            lip_deltadelta = (lip_deltadelta * lip_std) + lip_mean
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

        if cfg.model.gray:
            lip_orig = lip_orig.expand(-1, -1, -1, 3)
            lip_delta = lip_delta.expand(-1, -1, -1, 3)
            lip_deltadelta = lip_deltadelta.expand(-1, -1, -1, 3)

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
    else:
        lip_orig = lip

        # 標準化したので元のスケールに直す
        if lip_std.dim() == 1:
            lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
            lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
            lip_orig = torch.mul(lip_orig, lip_std)
            lip_orig = torch.add(lip_orig, lip_mean)
        elif lip_std.dim() == 3:
            lip_std = lip_std.unsqueeze(-1)     # (C, H, W, 1)
            lip_mean = lip_mean.unsqueeze(-1)   # (C, H, W, 1)
            lip_orig = torch.mul(lip_orig, lip_std)
            lip_orig = torch.add(lip_orig, lip_mean)
        
        lip_orig = lip_orig.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
        lip_orig = lip_orig.to('cpu')

        if cfg.model.gray:
            lip_orig = lip_orig.expand(-1, -1, -1, 3)

        torchvision.io.write_video(
            filename=str(save_path / "lip.mp4"),
            video_array=lip_orig,
            fps=cfg.model.fps
        )


def save_landmark_video(landmark, save_dir):
    landmark = landmark.to('cpu').numpy()
    fig_norm = plt.figure()
    image_norm_list = []
    for i in range(landmark.shape[0]):
        coords_x = landmark[i, 0, :]
        coords_y = landmark[i, 1, :]

        image_norm = plt.plot(coords_x, coords_y, linestyle="None", marker="o", color="c")
        plt.axis([
            np.min(coords_x) + (np.min(coords_x) / 5), 
            np.max(coords_x) + (np.max(coords_x) / 5), 
            np.max(coords_y) + (np.max(coords_y) / 5), 
            np.min(coords_y) + (np.min(coords_y) / 5),
        ])
        image_norm_list.append(image_norm)

    anime = animation.ArtistAnimation(fig_norm, image_norm_list, interval=20)
    os.makedirs(save_dir, exist_ok=True)
    anime.save(f"{str(save_dir)}/landmark.mp4")
    plt.close()


def calc_wav(cfg, save_path, file_name, feature, feat_mean, feat_std):
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
    time = np.arange(0, f0_input.shape[0]) / 100

    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    plt.plot(time, f0_input, label="input")
    plt.plot(time, f0_AbS, label="Analysis by Synthesis")
    plt.plot(time, f0_gen, label="Synthesis")
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.2)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[Hz]")
    plt.title("f0")
    plt.grid()

    # ax = plt.subplot(3, 1, 1)
    # ax.plot(time, f0_input)
    # plt.xlabel("Time[s]")
    # plt.ylabel("f0[hz]")
    # plt.title("Input")
    # plt.grid()

    # ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    # ax.plot(time, f0_AbS)
    # plt.xlabel("Time[s]")
    # plt.ylabel("f0[hz]")
    # plt.title("Analysis by Synthesis")
    # plt.grid()

    # ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    # ax.plot(time, f0_gen)
    # plt.xlabel("Time[s]")
    # plt.ylabel("f0[hz]")
    # plt.title("Synthesis")
    # plt.grid()

    # plt.tight_layout()
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
    
    time = np.arange(0, f0_input.shape[0]) / 100

    plt.close("all")
    plt.figure()

    plt.plot(time, f0_input, label="input")
    # plt.plot(time, f0_AbS, label="Analysis by Synthesis")
    plt.plot(time, f0_gen, label="Synthesis")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.2)
    plt.xlabel("Time[s]")
    plt.ylabel("f0[Hz]")
    plt.title("f0")
    plt.grid()
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


def save_data(cfg, save_path, wav, lip, feature, feat_add, output, lip_mean, lip_std, feat_mean, feat_std):
    """
    出力データの保存
    """
    wav = wav.squeeze(0)
    lip = lip.squeeze(0)
    feature = feature.squeeze(0)
    feat_add = feat_add.squeeze(0)
    output = output.squeeze(0)

    wav = wav.to('cpu').numpy()

    save_lip_video(
        cfg=cfg,
        save_path=save_path,
        lip=lip,
        lip_mean=lip_mean,
        lip_std=lip_std
    )

    wav_AbS = calc_wav(
        cfg=cfg,
        save_path=save_path,
        file_name="AbS",
        feature=feature,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )

    wav_gen = calc_wav(
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

    write(str(save_path / "input.wav"), rate=cfg.model.sampling_rate, data=wav)
    write(str(save_path / "abs.wav"), rate=cfg.model.sampling_rate, data=wav_AbS)
    write(str(save_path / "generate.wav"), rate=cfg.model.sampling_rate, data=wav_gen)

    # プロット
    plot_wav(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_mel(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_spec(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_f0_from_wav(cfg, save_path, wav, wav_AbS, wav_gen)

    
def save_data_lipreading(cfg, save_path, target, output, classes_index):
    """
    target : (T,)
    output : (C, T)
    """
    target = target.to("cpu").detach().numpy()
    output = output.to("cpu").detach().numpy()

    phoneme_answer = [get_keys_from_value(classes_index, i) for i in target]
    phoneme_answer = " ".join(phoneme_answer)

    # 予測結果にはeosが連続する場合があるので、除去する
    phoneme_predict = [get_keys_from_value(classes_index, i) for i in output]
    first_eos_index = 0
    for i in range(len(phoneme_predict)):
        if phoneme_predict[i] == "eos":
            first_eos_index = i + 1
            break
    phoneme_predict = phoneme_predict[:first_eos_index]
    phoneme_predict = " ".join(phoneme_predict)

    phoneme_error_rate = wer(phoneme_answer, phoneme_predict)

    with open(str(save_path / "phoneme.txt"), "a") as f:
        f.write("answer\n")
        f.write(f"{phoneme_answer}\n")
        f.write("\npredict\n")
        f.write(f"{phoneme_predict}\n")
        f.write(f"\nphoneme error rate = {phoneme_error_rate}\n")

    return phoneme_error_rate


def save_data_pwg(cfg, save_path, target, output, ana_syn=None):
    target = target.squeeze(0)
    output = output.squeeze(0).squeeze(0)
    target = target.to('cpu').detach().numpy()
    output = output.to('cpu').detach().numpy()
    target /= np.max(np.abs(target))
    output /= np.max(np.abs(output))
    target = target.astype(np.float32)
    output = output.astype(np.float32)

    write(str(save_path / "input.wav"), rate=cfg.model.sampling_rate, data=target)
    write(str(save_path / "generate.wav"), rate=cfg.model.sampling_rate, data=output)

    if ana_syn is not None:
        ana_syn = ana_syn.squeeze(0).squeeze(0)
        ana_syn = ana_syn.to('cpu').detach().numpy()
        ana_syn /= np.max(np.abs(ana_syn))
        ana_syn = ana_syn.astype(np.float32)
        write(str(save_path / "abs.wav"), rate=cfg.model.sampling_rate, data=ana_syn)

    target = wav2mel(target, cfg, ref_max=True)
    output = wav2mel(output, cfg, ref_max=True)
    
    plt.close("all")
    plt.figure()
    ax = plt.subplot(2, 1, 1)
    specshow(
        data=target, 
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
    plt.title("target")
    
    ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=output, 
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
    plt.title("output")

    plt.tight_layout()
    plt.savefig(str(save_path / "mel.png"))


def save_data_tts(cfg, save_path, wav, feature, output, feat_mean, feat_std):
    wav = wav.squeeze(0)
    feature = feature.squeeze(0)
    output = output.squeeze(0)

    wav = wav.to('cpu').numpy()

    wav_AbS = calc_wav(
        cfg=cfg,
        save_path=save_path,
        file_name="AbS",
        feature=feature,
        feat_mean=feat_mean,
        feat_std=feat_std,
    )

    wav_gen = calc_wav(
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

    write(str(save_path / "input.wav"), rate=cfg.model.sampling_rate, data=wav)
    write(str(save_path / "abs.wav"), rate=cfg.model.sampling_rate, data=wav_AbS)
    write(str(save_path / "generate.wav"), rate=cfg.model.sampling_rate, data=wav_gen)

    # プロット
    plot_wav(cfg, save_path, wav, wav_AbS, wav_gen)
    plot_mel(cfg, save_path, wav, wav_AbS, wav_gen)


def visualize_feature_map_video(feature_map, save_dir, mean_or_max):
    """
    3次元畳み込みから得られる特徴マップの可視化
    feature_map : (B, C, T, H, W)
    """
    print("visualize video")
    feature_map = feature_map[0, ...]
    if mean_or_max == "max":
        feature_map, _ = torch.max(feature_map, dim=0, keepdim=False)    # (T, H, W)
    elif mean_or_max == "mean":
        feature_map = torch.mean(feature_map, dim=0, keepdim=False)    # (T, H, W)

    feature_map = feature_map.to("cpu").detach().numpy()

    # 最初と最後のフレームはなんか変なので省略
    # 畳み込みの時に0パディングしているので、その影響でちょっと変なのかも
    feature_map = feature_map[1:-1, ...]

    fig, ax = plt.subplots()

    def init():
        sns.heatmap(
            np.zeros_like(feature_map[0, ...]), 
            cbar=True, 
            vmin=np.min(feature_map), 
            vmax=np.max(feature_map), 
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
        )

    def update(frame):
        sns.heatmap(
            feature_map[frame, ...], 
            cbar=False, 
            vmin=np.min(feature_map), 
            vmax=np.max(feature_map), 
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
        )

    anime = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=range(feature_map.shape[0]),
        init_func=init,
        interval=20,
        repeat=False,
    )
    anime.save(str(save_dir / f"size_{feature_map.shape[1]}.mp4"))
    plt.close()


def visualize_feature_map_image(feature_map, save_dir, mean_or_max):
    print("visualize image")
    feature_map = feature_map[0, ...]
    if mean_or_max == "max":
        feature_map, _ = torch.max(feature_map, dim=0, keepdim=False)    # (T, H, W)
    elif mean_or_max == "mean":
        feature_map = torch.mean(feature_map, dim=0, keepdim=False)    # (T, H, W)

    feature_map = feature_map.to("cpu").detach().numpy()

    # 最初と最後のフレームはなんか変なので省略
    # 畳み込みの時に0パディングしているので、その影響でちょっと変なのかも
    feature_map = feature_map[1:-1, ...]

    img_save_dir = save_dir / f"size_{feature_map.shape[1]}"
    img_save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(feature_map.shape[0]):
        plt.figure()
        sns.heatmap(
            feature_map[i, ...], 
            cbar=True, 
            vmin=np.min(feature_map), 
            vmax=np.max(feature_map), 
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
        )
        plt.savefig(str(img_save_dir / f"{i}.png"))
        plt.close()