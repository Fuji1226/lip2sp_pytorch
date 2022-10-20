import matplotlib.pyplot as plt
from librosa.display import specshow
import numpy as np
from scipy.io.wavfile import write
from data_process.feature import wave2mel


def save_wav(save_path, filename, fs, wav):
    print(f"save_path = {save_path}")
    print(f"filename = {filename}")
    print(f"fs = {fs}")
    print(f"wav = {wav.shape}")
    save_path = save_path / filename
    write(str(save_path), fs, wav)


def save_waveform(save_path, filename, wav, wav_gen, fs):
    save_path = save_path / filename

    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    duration = wav.shape[0] / fs
    time = np.linspace(0, duration, wav.shape[0])

    ax = plt.subplot(2, 1, 1)
    ax.plot(time, wav)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.title("input")

    ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    ax.plot(time, wav_gen)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.title("output")

    plt.grid()
    plt.tight_layout()
    plt.savefig(str(save_path))


def save_mel(save_path, filename, wav, wav_gen, cfg):
    save_path = save_path / filename

    mel_input = wave2mel(
        wave=wav,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        n_mels=cfg.model.n_mel_channels,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
    )
    mel_gen = wave2mel(
        wave=wav_gen,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        n_mels=cfg.model.n_mel_channels,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
    )

    plt.close("all")
    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)

    ax = plt.subplot(2, 1, 1)
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
    plt.title("input")
    # plt.tick_params(labelbottom=False)
    
    ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
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
    plt.savefig(str(save_path))


def save_data(cfg, wav, wav_gen, save_path):
    fs = cfg.model.sampling_rate

    # wavファイルの保存
    save_wav(save_path, "input.wav", fs, wav)
    save_wav(save_path, "synthesis.wav", fs, wav_gen)

    # 音声波形の保存
    save_waveform(save_path, "waveform.png", wav, wav_gen, fs)

    # メルスペクトログラムの保存
    save_mel(save_path, "mel.png", wav, wav_gen, cfg)
