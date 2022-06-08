"""
reference
https://github.com/Chris10M/Lip2Speech.git

一旦データが出るようになりました
"""

import os
import sys

from matplotlib.pyplot import step
from yaml import load
# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from email.mime import audio
from multiprocessing.sharedctypes import Value
import os
import random
from pathlib import Path

# brew install ffmpeg -> pip install ffmpeg-python
import ffmpeg   
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchaudio
from torch.utils.data import Dataset, DataLoader, get_worker_info
import librosa

# 自作
from data_process.audio_process import MelSpectrogram
from get_dir import get_datasetroot, get_data_directory
from hparams import create_hparams
from transform import preprocess, load_data


data_root = Path(get_datasetroot()).expanduser()    

hparams = create_hparams()


def get_datasets(data_root, mode):
    """
    train用とtest用のデータのディレクトリを分けておいて、modeで分岐させる感じにしてみました
    とりあえず
    dataset/lip/lip_cropped         にtrain用データ
    dataset/lip/lip_cropped_test    にtest用データを適当に置いてやってみて動きました
    """
    items = dict()
    idx = 0
    if mode == "train":
        for curDir, Dir, Files in os.walk(data_root):
            for filename in Files:
                # curDirの末尾がlip_croppedの時
                if curDir.endswith("lip_cropped"):
                    # filenameの末尾（拡張子）が.mp4の時
                    if filename.endswith(".mp4"):
                        format = ".mp4"
                        video_path = os.path.join(curDir, filename)
                        # 名前を同じにしているので拡張子だけ.wavに変更
                        audio_path = os.path.join(curDir, filename.replace(str(format), ".wav"))

                        if os.path.isfile(audio_path) and os.path.isfile(audio_path):
                            items[idx] = [video_path, audio_path]
                            idx += 1
                else:
                    continue
    else:
        for curDir, Dir, Files in os.walk(data_root):
            for filename in Files:
                # curDirの末尾がlip_cropped_testの時
                if curDir.endswith("lip_cropped_test"):
                    # filenameの末尾（拡張子）が.mp4の時
                    if filename.endswith(".mp4"):
                        format = ".mp4"
                        video_path = os.path.join(curDir, filename)
                        # 名前を同じにしているので拡張子だけ.wavに変更
                        audio_path = os.path.join(curDir, filename.replace(str(format), ".wav"))

                        if os.path.isfile(audio_path) and os.path.isfile(audio_path):
                            items[idx] = [video_path, audio_path]
                            idx += 1
                else:
                    continue
    return items


# def normalization(data_root=data_root, mode='train'):
#     items = get_datasets(data_root, mode)
#     data_len = len(items)
#     item_iter = iter(items)
#     item_idx = next(item_iter)

#     while item_idx:
#         video_path, audio_path = items[item_idx]


def calc_mean_var(items, len):
    mean = 0
    var = 0
    for i in range(len):
        video_path, audio_path = items[i]
        (lip, y, feat_add, upsample) = load_data(
            data_path=Path(video_path),
            gray=hparams.gray,
            frame_period=hparams.frame_period,
            feature_type=hparams.feature_type,
            nmels=hparams.n_mel_channels,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
        )

        # 時間方向に平均と分散を計算
        mean += np.mean(y, axis=0)
        var += np.var(y, axis=0)
    mean /= len
    var /= len

    return mean, var


class KablabDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        assert mode in ('train', 'test')
        self.data_root = data_root
        self.mode = mode

        # self.trans = transforms.Compose([
        #     transforms.Lambda(lambda im: im.float() / 255.0),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        # 口唇動画、音声データまでのパス一覧を取得
        self.items = get_datasets(self.data_root, self.mode)
        self.len = len(self.items)
        
        self.mean, self.var = calc_mean_var(self.items, self.len)
        
        print(f'Size of {type(self).__name__}: {self.len}')

        random.shuffle(self.items)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #item_idx = next(self.item_iter)
        video_path, audio_path = self.items[index]
        data_path = Path(video_path)

        # modeを追加
        # "train"でデータ拡張とか通るようになるように分岐
        ret, data_len = preprocess(
            data_path=data_path,
            gray=hparams.gray,
            delta=hparams.delta,
            frame_period=hparams.frame_period,
            feature_type=hparams.feature_type,
            nmels=hparams.n_mel_channels,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
            length=hparams.length,
            mean=self.mean,
            var=self.var,
            mode=self.mode,
        )

        return ret, data_len
        

def main():
    print("############################ Start!! ############################")
    datasets = KablabDataset(data_root, mode="train")
    # datasets = KablabDataset(data_root, mode="test")
    loader = DataLoader(
        dataset=datasets,
        batch_size=5,   
        shuffle=True,
        num_workers=0,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )

    # print(datasets.__len__())
    # print(datasets.current_item)
    # print(datasets.reset_item())

    # データ確認用。__getitem__(self, _)を__getitem__(self)に変更すれば見れます！
    # ただ、そうすると下のresultsの方は見れません。どっちかだけです。
    # frames, waveform, melspec = datasets.__getitem__()
    # print("####### type #######")
    # print(f"type(frames) = {type(frames)}")
    # print(f"type(waveform) = {type(waveform)}")
    # print(f"type(melspec) = {type(melspec)}")

    # print("\n####### shape #######")
    # print(f"frames.shape = {frames.shape}")
    # print(f"waveform.shape = {waveform.shape}")
    # print(f"melspec.shape = {melspec.shape}")

    # print("\n####### len #######")
    # print(f"len(frames) = {len(frames)}")
    # print(f"len(waveform) = {len(waveform)}")
    # print(f"len(melspec) = {len(melspec)}")


    # results
    for interation in range(hparams.max_iter):
        for bdx, batch in enumerate(loader):
            (lip, y, feat_add), data_len = batch
            print("################################################")
            print(type(lip))
            print(type(y))
            print(type(feat_add))
            print(f"lip = {lip.shape}")  # (B, C=5, W=48, H=48, T=150)
            print(f"y(acoustic features) = {y.shape}") # (B, C, T=300)
            print(f"feat_add = {feat_add.shape}")     # (B, C=3, T=300)
            print(f"data_len = {data_len}")






if __name__ == "__main__":
    # print(data_root)
    main()