from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import librosa
import random
import hydra


OVERLAP = 4
EPS = 1.0e-6
DEBUG = False
if DEBUG:
    SAVE_DIR = Path("~/dataset/lip/np_files/jvs_debug/train").expanduser()
else:
    SAVE_DIR = Path("~/dataset/lip/np_files/jvs/train").expanduser()


def load_jvs_path():
    jvs_dir = Path("~/dataset/jvs_ver1").expanduser()
    speaker_list = [f"jvs{i:03d}" for i in range(1, 101)]
    data_list = ["nonpara30", "parallel100"]
    data_path_list_all = []
    for speaker in speaker_list:
        for data in data_list:
            data_dir = jvs_dir / speaker / data / "wav24kHz16bit"
            data_path_list = list(data_dir.glob("*.wav"))
            data_path_list_all += data_path_list
    return data_path_list_all


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


@hydra.main(version_base=None, config_name="config", config_path="../../conf")
def main(cfg):
    data_path_list = load_jvs_path()
    if DEBUG:
        data_path_list = data_path_list[:10]
    
    for data_path in tqdm(data_path_list):
        save_dir = SAVE_DIR / data_path.parents[2].name / "mspec80"
        save_dir.mkdir(parents=True, exist_ok=True)
        if (save_dir / f"{data_path.stem}.npz").exists():
            continue
        
        wav, _ = librosa.load(str(data_path), sr=cfg.model.sampling_rate)
        wav /= np.max(np.abs(wav))
        feature = wav2mel(wav, cfg, ref_max=False)
        
        data_len = int(feature.shape[1] // 4 * 4)
        
        feature = feature[:, :data_len]
        feature = feature.T
        
        n_wav_sample_per_frame = cfg.model.sampling_rate * cfg.model.frame_period // 1000
        wav = wav[:int(n_wav_sample_per_frame * data_len)]
        wav_padded = np.zeros(int(n_wav_sample_per_frame * data_len))
        wav_padded[:wav.shape[0]] = wav
        wav = wav_padded
        
        lip = np.random.rand(1, 96, 96, data_len // 4)
        
        np.savez(
            file=str(save_dir / data_path.stem),
            wav=wav,
            lip=lip,
            feature=feature,
        )
        
    
    
if __name__ == "__main__":
    main()