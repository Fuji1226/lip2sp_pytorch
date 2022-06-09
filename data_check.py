"""
dataのtransform後の可視化用です
save_pathには自分の保存したいところを設定してください
"""

import torch
import torchvision
from data_process.feature import mel2wave, world2wav
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def data_check_trans(cfg, index, data, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
    ########################################################################################
    # 可視化
    ########################################################################################
    lip = data[0]   # (C, W, H, T)
    feature = data[1]   # (C, T)
    feat_add = data[2]  # (C, T)
    save_path = "/users/minami/dataset/after_trans"
    
    # 口唇動画
    lip = lip[:3, ...]
    
    # 正規化したので元のスケールに直す
    lip_std = lip_std.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    lip_mean = lip_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    lip = torch.mul(lip, lip_std)
    lip = torch.add(lip, lip_mean)
    lip = lip.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
    torchvision.io.write_video(
        filename=save_path+f"/lip_trans{index}.mp4",
        video_array=lip,
        fps=cfg.model.fps
    )
    ########################################################################################
    # 動的特徴量
    lip_delta = data[0][-2, ...]

    # 正規化したので元のスケールに直す
    lip_delta = torch.mul(lip_delta, lip_std)
    lip_delta = torch.add(lip_delta, lip_mean)

    lip_deltadelta = data[0][-1, ...]

    # 正規化したので元のスケールに直す
    lip_deltadelta = torch.mul(lip_deltadelta, lip_std)
    lip_deltadelta = torch.add(lip_deltadelta, lip_mean)

    lip_delta = lip_delta.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
    lip_deltadelta = lip_deltadelta.permute(-1, 1, 2, 0).to(torch.uint8)  # (T, W, H, C)
    
    torchvision.io.write_video(
        filename=save_path+f"/lip_delta_trans{index}.mp4",
        video_array=lip_delta,
        fps=cfg.model.fps
    )
    torchvision.io.write_video(
        filename=save_path+f"/lip_deltadelta_trans{index}.mp4",
        video_array=lip_deltadelta,
        fps=cfg.model.fps
    )
    ########################################################################################
    # feat_add
    feat_add = feat_add.to('cpu').detach().numpy().copy()
    feat_add_mean = feat_add_mean.unsqueeze(1).to('cpu').detach().numpy().copy()
    feat_add_std = feat_add_std.unsqueeze(1).to('cpu').detach().numpy().copy()

    # 正規化したので元のスケールに直す
    feat_add *= feat_add_std
    feat_add += feat_add_mean

    f0 = feat_add[0]
    vuv = feat_add[1]
    power = feat_add[2]

    x = np.arange(feat_add.shape[-1])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, f0)
    plt.savefig(save_path+f"/feat_add_f0{index}.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, vuv)
    plt.savefig(save_path+f"/feat_add_vuv{index}.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, power)
    plt.savefig(save_path+f"/feat_add_power{index}.png")

    ########################################################################################
    # メルスペクトログラム
    if cfg.model.feature_type == "mspec":

        feature = feature.to('cpu').detach().numpy().copy()
        feat_mean = feat_mean.unsqueeze(1).to('cpu').detach().numpy().copy()
        feat_std = feat_std.unsqueeze(1).to('cpu').detach().numpy().copy()

        # 正規化したので元のスケールに直す
        feature *= feat_std
        feature += feat_mean
        
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            data=feature,
            x_axis='time',
            y_axis='mel',
            sr=cfg.model.sampling_rate,
            fmax=cfg.model.f_max,
            fmin=cfg.model.f_min,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        plt.savefig(save_path+f"/mel_trans{index}.png")
        
        # メルスペクトログラムからgriffin-limによる音声合成
        audio = librosa.feature.inverse.mel_to_audio(
            librosa.db_to_power(feature),
            sr=cfg.model.sampling_rate,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
            n_iter=50,
        )
        write(save_path+f"/audio{index}.wav", rate=cfg.model.sampling_rate, data=audio)
    #########################################################################################
    # world特徴量
    if cfg.model.feature_type == "world":
        feature = feature.to('cpu').detach().numpy().copy()
        feat_mean = feat_mean.unsqueeze(1).to('cpu').detach().numpy().copy()
        feat_std = feat_std.unsqueeze(1).to('cpu').detach().numpy().copy()
        
        # 正規化したので元のスケールに直す
        feature *= feat_std
        feature += feat_mean

        feature = feature.T
        mcep = feature[:, :-3]
        clf0 = feature[:, -3]
        vuv = feature[:, -2]
        cap = feature[:, -1]
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
        write(save_path+f"/out_world{index}.wav", rate=cfg.model.sampling_rate, data=wav)

        # メルスペクトログラム
        S = librosa.feature.melspectrogram(
            y=wav, sr=cfg.model.sampling_rate, n_mels=cfg.model.n_mel_channels, fmax=cfg.model.f_max,
            n_fft=cfg.model.n_fft, hop_length=cfg.model.hop_length, win_length=cfg.model.win_length)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            data=S_dB,
            x_axis='time',
            y_axis='mel',
            sr=cfg.model.sampling_rate,
            fmax=cfg.model.f_max,
            fmin=cfg.model.f_min,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        plt.savefig(save_path+f"/mel_world{index}.png")

        # world特徴量の可視化
        x = np.arange(feature.shape[0])
        # メルケプストラム
        mcep = mcep.T
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            data=mcep, 
            sr=cfg.model.sampling_rate,
            x_axis='time',
            fmax=cfg.model.f_max,
            fmin=cfg.model.f_min,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='mel cepstrum')
        plt.savefig(save_path+f"/world_mcep{index}.png")

        # 連続対数F0
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, clf0)
        plt.savefig(save_path+f"/world_clf0{index}.png")

        # 有声/無声判定
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, vuv)
        plt.savefig(save_path+f"/world_vuv{index}.png")

        # 帯域非周期性指標
        # ここはうまくいってません…
        cap = cap[np.newaxis, :]
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            data=cap, 
            sr=cfg.model.sampling_rate,
            x_axis='time',
            fmax=cfg.model.f_max,
            fmin=cfg.model.f_min,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='aperiodicity')
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(x, cap)
        plt.savefig(save_path+f"/world_cap{index}.png")


    return