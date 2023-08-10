from pathlib import Path
import librosa
import numpy as np
import hydra
import random
import shutil

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


@hydra.main(version_base=None, config_name="config", config_path="../../conf")
def main(cfg):
    data_dir = Path("~/dataset/jsut_ver1.1").expanduser()
    data_dir_list = list(data_dir.glob("*"))
    data_dir_list = [d for d in data_dir_list if d.suffix != ".txt"]
    data_path_list = []
    for d in data_dir_list:
        d = d / "wav"
        data_path_list += list(d.glob("*.wav"))
        
    data_path_list = random.sample(data_path_list, 10)
    
    save_dir = Path("~/lip2sp_pytorch/data_process/jsut/test_sample").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    for data_path in data_path_list:
        shutil.copy(str(data_path), str(save_dir))
    
    
if __name__ == "__main__":
    main()