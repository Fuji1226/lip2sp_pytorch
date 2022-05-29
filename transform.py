"""
先輩のpreprocessを使って最後に型変換すればそのまま使えそうなので、とりあえずそうしてます。
"""

from skvideo.io import vread, vwrite
from pathlib import Path
import cv2
import numpy as np
import os
import torch
import torchvision
import librosa
from zipfile import BadZipFile

from scipy import signal
from scipy.interpolate import interp1d
import pyworld
from pyreaper import reaper
import pysptk
from pysptk import swipe

from get_dir import get_data_directory
from utils import get_sp_name, get_upsample
from data_process.feature import wave2mel, wav2world
from data_process.color_jitter import color_jitter
from hparams import create_hparams

# 一旦chainerを使って同じ処理を行い、最後にtensorに型変換する感じでやってます
from chainercv.transforms import random_crop, random_flip
import chainer
from chainer.functions import resize_images
from functools import partial
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate, zoom

ROOT = Path(get_data_directory())


def interpolate_1d(x, t):
    with chainer.no_backprop_mode():
        y = resize_images(x[None, None].astype("float32"),
                          (t, x.shape[-1]), mode="nearest", align_corners=True)
    return chainer.as_array(y).squeeze((0, 1))


def random_translate(img, ratio=0.1):

    if np.random.rand() < ratio:
        fillcolor = tuple(img.mean((0, 1)).astype(img.dtype).tolist())
        translate = np.random.randint(3, size=2).tolist()
        img = np.asarray(Image.fromarray(img).rotate(
            0, fillcolor=fillcolor, translate=translate))

    return img


def pad_mov_edge(mov, width_shift=4, height_shift=4):
    pad_width = ((0, 0), (width_shift, width_shift),
                 (height_shift, height_shift))
    if mov.ndim > len(pad_width):
        pad_width += tuple([(0, 0) for _ in range(mov.ndim - len(pad_width))])
    pad_func = partial(np.pad, pad_width=pad_width, mode="constant")

    mov = np.vstack([pad_func(arr, constant_values=arr.mean())
                     for arr in np.split(mov, mov.shape[0])])

    return mov


def random_shift(mov, width_shift=4, height_shift=4):
    assert mov.shape[0] in (1, 3)
    w, h = mov.shape[1:3]
    mov = pad_mov_edge(mov, width_shift, height_shift)
    return random_crop(mov, (w, h))


def random_rotate(mov, range=10):
    rng = np.abs(range)
    angle = np.random.uniform(-rng, rng)
    rotate_func = partial(
        rotate, angle=angle, axes=(2, 1),
        reshape=False, mode="constant"
    )
    mov = np.vstack(
        [rotate_func(arr, cval=arr.mean())
         for arr in np.split(mov, mov.shape[0])]
    )

    return mov


def random_zoom(mov, r):
    assert mov.shape[0] in (1, 3)
    w, h = mov.shape[1:3]
    r = min(r, 1)
    r = r + (1-r) * np.random.rand()
    factor = 1 / r - 1
    w_pad = int(w * factor / 2) + 1
    h_pad = int(h * factor / 2) + 1
    mov = pad_mov_edge(mov, w_pad, h_pad)
    mov = zoom(mov, (1, r, r, 1), order=0)
    return random_crop(mov, (w, h))


def lip_augument(lip):
    dtype = lip.dtype
    amin = np.iinfo(dtype).min
    amax = np.iinfo(dtype).max

    lip = random_flip(lip, x_random=True)
    lip = color_jitter(lip)
    lip = np.clip(lip, amin, amax).astype(dtype)

    lip = random_rotate(lip, range=10)

    if np.random.rand() > 0.5:
        lip = random_shift(lip)
    else:
        lip = random_zoom(lip, 0.8)

    if np.random.rand() < 0.5:
        lip = np.asarray([random_translate(frame) for frame in lip.T]).T

    return lip


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
    if path is not None:
        name = get_sp_name(path.stem, feature_type, frame_period, nmels)
        load_path = get_sp_path(name, path)

        try:
            y = np.load(load_path, mmap_mode="r", allow_pickle=False)
            loaded = True
        except (FileNotFoundError, BadZipFile):
            pass

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


def load_data(data_path, gray, frame_period, feature_type, nmels, f_min, f_max, mode=None):
    """
    先輩のコードはlabel, label_nameを取得していて、それをget_dvectorとかに使用している
    おそらく捜索した時のファイル名とかを取ってくる感じだと思うのですが、一旦スルーしてます
    lip2sp/submodules/mychainerutils/dataset.py 128行目からのPathDatasetクラスが関係してそうです
    """
    if mode == "check":
        print("######### loading data #########")
    lip, fps = load_mp4(str(data_path), gray)
    sppath = Path(data_path)
    sppath = sppath.parent / (sppath.stem + ".wav")
    wave, fs = librosa.load(str(sppath), sr=None, mono=None)
    wave = wave[:int(lip.shape[-1]/fps*1.2*fs)]

    if mode == "check":
        print(lip.shape)
        print(wave.shape)

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

    if mode == "check":
        print(y.shape)

    power = librosa.feature.rms(wave, frame_length=hop_length*2,
                hop_length=hop_length).squeeze()
    power = fill_nan(power)
    f0 = swipe(wave.astype("float64"), fs, hop_length,
               min=70.0, otype='f0').squeeze()

    if mode == "check":
        print(f"f0.size(before continuous_f0) = {f0.size}")

    f0, vuv = continuous_f0(f0)
    T = min(power.size, f0.size, y.shape[0])

    if mode == "check":
        print(f"power.size = {power.size}")
        print(f"f0.size = {f0.size}")
        print(f"y.shape[0] = {y.shape[0]}")

    feat_add = np.vstack((f0[:T], vuv[:T], power[:T])).T
    feat_add = np.log(np.maximum(feat_add, 1.0e-7))
    y = y[:T]

    if mode == "check":
        print(f"y.shape(after processing) = {y.shape}")

    ret = (
        lip, y, feat_add, upsample
    )

    return_wave = False

    if return_wave:
        return wave, fs
    else:
        return ret


def preprocess(
    data_path, gray, delta, frame_period, feature_type, nmels, f_min, f_max,
    length=None, mean=None, var=None, mode=None):

    # path = Path('/Users/minami/dataset/lip/lip_cropped/ATR503_j05_0.mp4')
    ret = load_data(data_path, gray, frame_period, feature_type, nmels, f_min, f_max, mode)
    # lip : (C, H, W, T)
    # y : (T, C)
    # feat_add : (T, C)
    # upsample : scalar
    (lip, y, feat_add, upsample) = ret

    if mode == "check":
        print("######### start #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")

    data_len = min(len(y) // upsample * upsample,  lip.shape[-1] * upsample)
    y = y[:data_len]
    feat_add = feat_add[:data_len]
    lip = lip[..., :data_len // upsample]

    if mode == "check":
        print("######### after first processing #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")

    # 学習時の処理
    if chainer.config.train:
        if mode == "check":
            print("augmentation...")
        lip = lip_augument(lip)

        """
        # plot lip
        from crop_movie import write_mp4
        write_mp4("lip.mp4", lip.swapaxes(0, -1)[..., ::-1], 25.0)
        import sys
        sys.exit()
        """
    # 学習時の処理
    if chainer.config.train:
        if mode == "check":
            print("processing...")
        rate = np.random.rand() * 0.5 + 1.
        T = y.shape[0]
        T_l = lip.shape[-1]
        idx = np.linspace(0, 1, int(T*rate) // upsample * upsample)
        idx = (idx - idx.min()) / (idx.max() - idx.min())
        idx_l = (idx[::upsample] * (T_l-1)).astype(int)
        lip = lip[..., idx_l]
        y = interpolate_1d(y, idx.size)
        feat_add = interpolate_1d(feat_add, idx.size)
        assert y.shape[0] == lip.shape[-1] * upsample

    if mode == "check":
        print("######### after second processing #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")

    data_len = min(len(y) // upsample * upsample,  lip.shape[-1] * upsample)
    y = y[:data_len]
    feat_add = feat_add[:data_len]
    lip = lip[..., :data_len // upsample]

    if mode == "check":
        print("######### after third processing #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")
        print(f"data_len = {data_len}")


    # hparamsではlength = 300になっている
    # 音響特徴量のフレーム数
    # length = 300
    if length:
        if data_len <= length:
            rep = length // data_len + 1
            y = np.tile(y, (rep, 1))
            feat_add = np.tile(feat_add, (rep, 1))
            lip = np.tile(lip, (1, 1, 1, rep))
            data_len = y.shape[0]
    mask = np.ones(data_len)

    if mode == "check":
        print("######### after fourth processing #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")
        print(f"mask = {mask.shape}")

    if length:
        length = length // upsample * upsample
        if data_len > length:
            index = np.random.randint(0, data_len - length) // upsample
            lip = lip[..., index:index + length // upsample]
            y = y[
                index * upsample:index * upsample + length]
            feat_add = feat_add[
                index * upsample:index * upsample + length]
            mask = mask[
                index * upsample:index * upsample + length]
    if mode == "check":
        print("######### after fifth processing #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")
        print(f"mask = {mask.shape}")
    
    # 音響特徴量の標準化
    # 事前に全データの音響特徴量から平均と分散を求めておいたらよさそう
    # mean = 0
    # var = 1
    if not (mean is None or var is None):
        y = (y - mean) / np.sqrt(var)

    lip = lip.astype('float32')
    y = y.T.astype('float32')
    feat_add = feat_add.T.astype('float32')
    mask = mask.astype('float32')

    # 動的特徴量の計算
    # delta = True
    if delta:
        lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]
        lip_pad = lip_pad.astype(lip.dtype)
        lip_pad = gaussian_filter(
            lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)
        lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
        lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
        lip_acc = lip_pad[..., 0:-2] + \
            lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
        lip = np.vstack((lip, lip_diff, lip_acc))

        """
        # plot lip
        from crop_movie import write_mp4
        lip_diff = (lip_diff - lip_diff.min()) / \
            (lip_diff.max() - lip_diff.min()) * 255
        lip_acc = (lip_acc - lip_acc.min()) / \
            (lip_acc.max() - lip_acc.min()) * 255
        write_mp4("lip_gray.mp4", lip_pad.swapaxes(0, -1), 25.0)
        write_mp4("lip_diff.mp4", lip_diff.swapaxes(0, -1), 25.0)
        write_mp4("lip_acc.mp4", lip_acc.swapaxes(0, -1), 25.0)
        import sys
        sys.exit()
        """
    if mode == "check":
        print("######### after sixth processing #########")
        print(f"lip = {lip.shape}")
        print(f"y = {y.shape}")
        print(f"feat_add = {feat_add.shape}")
        print(f"upsample = {upsample}")
        print(f"mask = {mask.shape}")

    lip = torch.from_numpy(lip)
    y = torch.from_numpy(y)
    mask = torch.from_numpy(mask)
    feat_add = torch.from_numpy(feat_add)
    # labelがわからないので一旦スルー
    # ret = (lip, y, mask, feat_add, label)
    ret = (lip, y, mask, feat_add)
    # if d_vectors is not None:
    #     d_vector = get_dvector(d_vectors, label_name).astype('float32')
    #     ret += (d_vector)

    return ret




# check
if __name__ == "__main__":
    data_path = Path('/Users/minami/dataset/lip/lip_cropped/ATR503_j05_0.mp4')
    hparams = create_hparams()

    ret = preprocess(
        data_path=data_path,
        gray=hparams.gray,
        delta=hparams.delta,
        frame_period=hparams.frame_period,
        feature_type=hparams.feature_type,
        nmels=hparams.n_mel_channels,
        f_min=hparams.f_min,
        f_max=hparams.f_max,
        length=hparams.length,
        mean=0,
        var=0,
        mode=None,  # "check"に変更すると色々出ます
    )
    print("Done!")
    print(type(ret[0]))
    print(type(ret[1]))
    print(type(ret[2]))
    print(type(ret[3]))
    print(ret[1].shape)
    