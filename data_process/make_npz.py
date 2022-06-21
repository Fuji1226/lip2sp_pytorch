"""
事前にnpzファイルを作るためのコード
F01_kablabなど話者ごとにディレクトリを分けて，その中にmp4とwavがあることを想定しています
また，学習用データとテスト用データも分かれていることを想定しています

dataset/lip/lip_cropped/train/F01_kablab
dataset/lip/lip_cropped/test/F01_kablab
こんな感じです
"""

from email.mime import audio
import os
import sys
import glob

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from tqdm import tqdm

import hydra

from transform_no_chainer import load_data_for_npz


def get_dataset(data_root):    
    """
    mp4, wavまでのパス取得
    mp4とwavが同じディレクトリに入っている状態を想定
    """
    train_items = []
    test_items = []
    print(f"data_root = {data_root}")
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".wav"):
                if Path(file).stem == "BASIC5000_0944_0":
                    print("----- file -----")
                    audio_path = os.path.join(curdir, file)
                    video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
                    print(audio_path)
                    print(video_path)
                    if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            train_items.append([video_path, audio_path])

                if '_j' in Path(file).stem:
                    audio_path = os.path.join(curdir, file)
                    video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
                    if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            test_items.append([video_path, audio_path])
                else:
                    audio_path = os.path.join(curdir, file)
                    video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
                    if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            train_items.append([video_path, audio_path])

            # if file.endswith(".mp4"):
            #     if '_j' in Path(file).stem:
            #         format = ".mp4"
            #         video_path = os.path.join(curdir, file)
            #         audio_path = os.path.join(curdir, file.replace(str(format), ".wav"))
            #         if os.path.isfile(video_path) and os.path.isfile(audio_path):
            #                 test_items.append([video_path, audio_path])
            #     else:
            #         format = ".mp4"
            #         video_path = os.path.join(curdir, file)
            #         audio_path = os.path.join(curdir, file.replace(str(format), ".wav"))
            #         if os.path.isfile(video_path) and os.path.isfile(audio_path):
            #                 train_items.append([video_path, audio_path])
    return train_items, test_items


def save_data_train(items, len, cfg, data_save_path, mean_std_save_path, device):
    """
    データ，平均，標準偏差の保存
    話者ごとに行うことを想定してます
    """
    lip_mean = 0
    lip_std = 0
    feat_mean = 0
    feat_std = 0
    feat_add_mean = 0
    feat_add_std = 0

    for i in range(len):
        video_path, audio_path = items[i]
        video_path, audio_path = Path(video_path), Path(audio_path)

        # 話者ラベル(F01_kablabとかです)
        speaker = audio_path.parents[0].name

        (lip, feature, feat_add, upsample), data_len = load_data_for_npz(
            video_path=video_path,
            audio_path=audio_path,
            cfg=cfg,
            gray=cfg.model.gray,
            frame_period=cfg.model.frame_period,
            feature_type=cfg.model.feature_type,
            nmels=cfg.model.n_mel_channels,
            f_min=cfg.model.f_min,
            f_max=cfg.model.f_max,
        )
        
        # データの保存
        os.makedirs(os.path.join(data_save_path, speaker), exist_ok=True)
        np.savez(
            f"{data_save_path}/{speaker}/{audio_path.stem}_{cfg.model.name}",
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            upsample=upsample,
            data_len=data_len,
        )

        lip = torch.from_numpy(lip).to(device)
        feature = torch.from_numpy(feature).to(device)
        feat_add = torch.from_numpy(feat_add).to(device)

        lip_mean += torch.mean(lip.float(), dim=(1, 2, 3))
        lip_std += torch.mean(lip.float(), dim=(1, 2, 3))
        feat_mean += torch.mean(feature, dim=0)
        feat_std += torch.std(feature, dim=0)
        feat_add_mean += torch.mean(feat_add, dim=0)
        feat_add_std += torch.std(feat_add, dim=0)

    # データ全体の平均、分散を計算 (C,) チャンネルごと
    lip_mean /= len     
    lip_std /= len      
    feat_mean /= len    
    feat_std /= len     
    feat_add_mean /= len
    feat_add_std /= len

    lip_mean = lip_mean.to('cpu').detach().numpy().copy()
    lip_std = lip_std.to('cpu').detach().numpy().copy()
    feat_mean = feat_mean.to('cpu').detach().numpy().copy()
    feat_std = feat_std.to('cpu').detach().numpy().copy()
    feat_add_mean = feat_add_mean.to('cpu').detach().numpy().copy()
    feat_add_std = feat_add_std.to('cpu').detach().numpy().copy()
    
    os.makedirs(os.path.join(mean_std_save_path, speaker), exist_ok=True)
    np.savez(
        f"{mean_std_save_path}/{speaker}/train_{cfg.model.name}",
        lip_mean=lip_mean, 
        lip_std=lip_std, 
        feat_mean=feat_mean, 
        feat_std=feat_std, 
        feat_add_mean=feat_add_mean, 
        feat_add_std=feat_add_std,
    )


def save_data_test(items, len, cfg, data_save_path, mean_std_save_path, device):
    lip_mean = 0
    lip_std = 0
    feat_mean = 0
    feat_std = 0
    feat_add_mean = 0
    feat_add_std = 0

    for i in range(len):
        video_path, audio_path = items[i]
        video_path, audio_path = Path(video_path), Path(audio_path)

        # 話者ラベル
        speaker = video_path.parents[0].name

        (lip, feature, feat_add, upsample), data_len = load_data_for_npz(
            video_path=video_path,
            audio_path=audio_path,
            cfg=cfg,
            gray=cfg.model.gray,
            frame_period=cfg.model.frame_period,
            feature_type=cfg.model.feature_type,
            nmels=cfg.model.n_mel_channels,
            f_min=cfg.model.f_min,
            f_max=cfg.model.f_max,
        )
        
        # データの保存
        os.makedirs(os.path.join(data_save_path, speaker), exist_ok=True)
        np.savez(
            f"{data_save_path}/{speaker}/{audio_path.stem}_{cfg.model.name}",
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            upsample=upsample,
            data_len=data_len,
        )

        lip = torch.from_numpy(lip).to(device)
        feature = torch.from_numpy(feature).to(device)
        feat_add = torch.from_numpy(feat_add).to(device)

        lip_mean += torch.mean(lip.float(), dim=(1, 2, 3))
        lip_std += torch.mean(lip.float(), dim=(1, 2, 3))
        feat_mean += torch.mean(feature, dim=0)
        feat_std += torch.std(feature, dim=0)
        feat_add_mean += torch.mean(feat_add, dim=0)
        feat_add_std += torch.std(feat_add, dim=0)

    # データ全体の平均、分散を計算 (C,) チャンネルごと
    lip_mean /= len     
    lip_std /= len      
    feat_mean /= len    
    feat_std /= len     
    feat_add_mean /= len
    feat_add_std /= len

    lip_mean = lip_mean.to('cpu').detach().numpy().copy()
    lip_std = lip_std.to('cpu').detach().numpy().copy()
    feat_mean = feat_mean.to('cpu').detach().numpy().copy()
    feat_std = feat_std.to('cpu').detach().numpy().copy()
    feat_add_mean = feat_add_mean.to('cpu').detach().numpy().copy()
    feat_add_std = feat_add_std.to('cpu').detach().numpy().copy()
    
    os.makedirs(os.path.join(mean_std_save_path, speaker), exist_ok=True)
    np.savez(
        f"{mean_std_save_path}/{speaker}/test_{cfg.model.name}",
        lip_mean=lip_mean, 
        lip_std=lip_std, 
        feat_mean=feat_mean, 
        feat_std=feat_std, 
        feat_add_mean=feat_add_mean, 
        feat_add_std=feat_add_std,
    )


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    train_items, test_items = get_dataset(
        data_root=cfg.train.train_path,
    )
    n_data_train = len(train_items)
    n_data_test = len(test_items)

    save_data_train(
        items=train_items,
        len=n_data_train,
        cfg=cfg,
        data_save_path=cfg.train.train_pre_loaded_path,
        mean_std_save_path=cfg.train.train_mean_std_path,
        device=device,
    )

    save_data_test(
        items=test_items,
        len=n_data_test,
        cfg=cfg,
        data_save_path=cfg.test.pre_loaded_path,
        mean_std_save_path=cfg.test.mean_std_path,
        device=device,
    )


def test():
    data_root = "/users/minami/dataset/lip/lip_cropped"
    items = []
    for curdir, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mp4'):
                # jセットだけ分けたい
                if '_j' in Path(file).stem:
                    continue
                else:
                    items.append(os.path.join(curdir, file))
    print(items)

if __name__ == "__main__":
    main()
    # test()