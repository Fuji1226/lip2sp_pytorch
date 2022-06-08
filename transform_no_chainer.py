"""
load_dataのみに変更

とりあえずchainerなくせました
"""

from skvideo.io import vread
from pathlib import Path
import cv2
import numpy as np
import torch
import librosa
from zipfile import BadZipFile
from scipy.interpolate import interp1d
from pysptk import swipe

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
            y = wave2mel(wave, fs, frame_period,
                         n_mels=nmels, fmin=f_min, fmax=f_max).T

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
    movie = cv2.VideoCapture(str(path))
    fps = int(movie.get(cv2.CAP_PROP_FPS))
    movie.release()

    mov = vread(str(path), as_grey=gray, outputdict={"-s": "48x48"})
    mov = np.asarray(mov).swapaxes(0, -1)

    return mov, fps


def load_data(data_path, gray, frame_period, feature_type, nmels, f_min, f_max, mode=None, delta=True):
    """
    先輩のコードはlabel, label_nameを取得していて、それをget_dvectorとかに使用している
    おそらく捜索した時のファイル名とかを取ってくる感じだと思うのですが、一旦スルーしてます
    lip2sp/submodules/mychainerutils/dataset.py 128行目からのPathDatasetクラスが関係してそうです

    これは一旦使えそう
    """
    lip, fps = load_mp4(str(data_path), gray)
    sppath = Path(data_path)
    sppath = sppath.parent / (sppath.stem + ".wav")
    wave, fs = librosa.load(str(sppath), sr=None, mono=None)
    wave = wave[:int(lip.shape[-1]/fps*1.2*fs)]
    # frame_period = 10
    # feature_type = "mspec"
    # nmels = 80
    # f_min = 0
    # f_max = 7600
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

    lip = torch.from_numpy(lip)
    y = torch.from_numpy(y)
    feat_add = torch.from_numpy(feat_add)
    data_len = torch.tensor(data_len)
    upsample = torch.tensor(upsample)

    ret = (
        lip, y, feat_add, upsample
    )

    return_wave = False

    if return_wave:
        return wave, fs
    else:
        return ret, data_len





# check
if __name__ == "__main__":
    data_path = Path('/Users/minami/dataset/lip/lip_cropped/ATR503_j05_0.mp4')
    hparams = create_hparams()