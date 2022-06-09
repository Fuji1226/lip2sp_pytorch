"""


"""

import wave
import skvideo
from skvideo.io import vread
from pathlib import Path
import cv2
import numpy as np
import torch
import librosa
from zipfile import BadZipFile
from scipy.interpolate import interp1d
from pysptk import swipe
import torchvision

try:
    from get_dir import get_data_directory
    from utils import get_sp_name, get_upsample
    from data_process.feature import wave2mel, wav2world
    from hparams import create_hparams
except:
    from .get_dir import get_data_directory
    from .utils import get_sp_name, get_upsample
    from .data_process.feature import wave2mel, wav2world
    from .hparams import create_hparams


ROOT = Path(get_data_directory())


def get_sp_path(name, path, data_root=ROOT, save_root=None, save_dir="sp"):
    relpath = Path(path).relative_to(data_root)

    if save_root is None:
        save_root = data_root

    world_dir = Path(save_root) / save_dir
    ret_path = world_dir.joinpath(
        *relpath.parent.parts[1:]) / (name + ".npy")

    return ret_path


def cals_sp(wave, fs, frame_period, feature_type, path=None, nmels=None, f_min=None, f_max=None):

    loaded = False
    # if path is not None:
    #     name = get_sp_name(path.stem, feature_type, frame_period, nmels)
    #     load_path = get_sp_path(name, path)

    #     try:
    #         y = np.load(load_path, mmap_mode="r", allow_pickle=False)
    #         loaded = True
    #     except (FileNotFoundError, BadZipFile):
    #         pass
    
    if not loaded:
        if feature_type == "mspec":
            # 対数メルスペクトログラム
            y = wave2mel(wave, fs, frame_period,
                         n_mels=nmels, fmin=f_min, fmax=f_max).T    # (T, C)

        elif feature_type == "world":
            # mcep, clf0, vuv, cap, _, fbin, _
            # 元々こうなっていましたが、wav2worldの引数が6つでバグったので合わせました。
            # 先輩のが謎です
            mcep, clf0, vuv, cap, fbin, _ = wav2world(
                wave, fs, frame_period=frame_period)
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


def load_data(data_path, gray, frame_period, feature_type, nmels, f_min, f_max, mode=None, delta=True, return_wave=False):
    """
    先輩のコードはlabel, label_nameを取得していて、それをget_dvectorとかに使用している
    おそらく捜索した時のファイル名とかを取ってくる感じだと思うのですが、一旦スルーしてます
    lip2sp/submodules/mychainerutils/dataset.py 128行目からのPathDatasetクラスが関係してそうです

    これは一旦使えそう
    """
    lip, fps = load_mp4(str(data_path), gray)   # lipはtensor
    sppath = Path(data_path)
    sppath = sppath.parent / (sppath.stem + ".wav")
    wave, fs = librosa.load(str(sppath), sr=None, mono=None)
    wave = wave[:int(lip.shape[-1]/fps*1.2*fs)]
    upsample = get_upsample(fps, fs, frame_period)
    
    # 音響特徴量への変換
    y = cals_sp(
        wave, fs, frame_period, feature_type,
        path=data_path, nmels=nmels, f_min=f_min, f_max=f_max)
    hop_length = fs * frame_period // 1000  # 160
    
    power = librosa.feature.rms(wave, frame_length=hop_length*2,
                hop_length=hop_length).squeeze()
    power = fill_nan(power)
    f0 = swipe(wave.astype("float64"), fs, hop_length,
               min=70.0, otype='f0').squeeze()

    f0, vuv = continuous_f0(f0)
    T = min(power.size, f0.size, y.shape[0])
    
    feat_add = np.vstack((f0[:T], vuv[:T], power[:T])).T
    feat_add = np.log(np.maximum(feat_add, 1.0e-7))
    y = y[:T]

    data_len = min(len(y) // upsample * upsample,  lip.shape[-1] * upsample)
    y = y[:data_len]
    feat_add = feat_add[:data_len]
    lip = lip[..., :data_len // upsample]

    # lip = torch.from_numpy(lip)
    y = torch.from_numpy(y)
    feat_add = torch.from_numpy(feat_add)
    data_len = torch.tensor(data_len)
    upsample = torch.tensor(upsample)

    ret = (
        lip, y, feat_add, upsample
    )

    if return_wave:
        return ret, data_len, wave
    else:
        return ret, data_len



########################################################################################################
# 可視化用
import hydra
from data_process.feature import mel2wave, world2wav
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa.display

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    video_path = "/users/minami/dataset/train/atr503_j01_0.mp4"
    save_path = "/users/minami/dataset/after_load_data"

    # 使うときはload_dataの返り値にwaveを追加してください
    data, data_len, wave = load_data(
        data_path=Path(video_path),
        gray=cfg.model.gray,
        frame_period=cfg.model.frame_period,
        feature_type=cfg.model.feature_type,
        nmels=cfg.model.n_mel_channels,
        f_min=cfg.model.f_min,
        f_max=cfg.model.f_max,
        return_wave=True
    )
    lip = data[0]
    feature = data[1]
    feat_add = data[2]
    lip = lip.to('cpu').detach().numpy().copy()
    feature = feature.to('cpu').detach().numpy().copy()
    feat_add = feat_add.to('cpu').detach().numpy().copy()

    feature = feature.transpose(1, 0)

    # 原音声
    write(save_path+f"/original.wav", rate=cfg.model.sampling_rate, data=wave)
    
    if cfg.model.feature_type == "mspec":
        # 音声合成
        wav = mel2wave(feature, cfg.model.sampling_rate, cfg.model.frame_period)
        write(save_path+f"/mel2wav.wav", rate=cfg.model.sampling_rate, data=wav)
        
        # メルスペクトログラム
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
        plt.savefig(save_path+"/mel.png")

        # 原音声から求めたメルスペクトログラム
        S = librosa.feature.melspectrogram(
            y=wave, sr=cfg.model.sampling_rate, n_mels=cfg.model.n_mel_channels, fmax=cfg.model.f_max,
            n_fft=cfg.model.n_fft, hop_length=cfg.model.hop_length, win_length=cfg.model.win_length)
        S = S[:, :-1]
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
        plt.savefig(save_path+"/mel_LIBROSA.png")

        # 原音声から求めたメルスペクトログラムから、音声合成
        audio = librosa.feature.inverse.mel_to_audio(
            librosa.db_to_power(S_dB),
            sr=16000,
            n_fft=cfg.model.n_fft,
            hop_length=cfg.model.hop_length,
            win_length=cfg.model.win_length,
            n_iter=50,
        )
        write(save_path+"/out_librosa.wav", rate=cfg.model.sampling_rate, data=audio)

    elif cfg.model.feature_type == "world":
        # 音声合成
        data, data_len, wave = load_data(
            data_path=Path(video_path),
            gray=cfg.model.gray,
            frame_period=cfg.model.frame_period,
            feature_type=cfg.model.feature_type,
            nmels=cfg.model.n_mel_channels,
            f_min=cfg.model.f_min,
            f_max=cfg.model.f_max,
            return_wave=True
        )
        feature = data[1].to('cpu').detach().numpy().copy()
        
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
        write(save_path+"/out_world.wav", rate=cfg.model.sampling_rate, data=wav)

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
        plt.savefig(save_path+"/mel_world.png")

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
        plt.savefig(save_path+"/world_mcep.png")

        # 連続対数F0
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, clf0)
        plt.savefig(save_path+"/world_clf0.png")

        # 有声/無声判定
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, vuv)
        plt.savefig(save_path+"/world_vuv.png")

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
        plt.savefig(save_path+"/world_cap.png")
########################################################################################################


# check
if __name__ == "__main__":
    main()