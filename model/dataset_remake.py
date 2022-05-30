"""
reference
https://github.com/Chris10M/Lip2Speech.git

一旦データが出るようになりました
"""

import os
import sys

from matplotlib.pyplot import step
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
from importlib_metadata import files
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchaudio
from torch.utils.data import Dataset, DataLoader, get_worker_info

# 自作
from data_process.audio_process import MelSpectrogram
from get_dir import get_datasetroot, get_data_directory
from hparams import create_hparams
from transform import preprocess


data_root = Path(get_datasetroot()).expanduser()    # 確認用

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


def x_round(x):
    return math.floor(x * 4) / 4


class KablabDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        assert mode in ('train', 'test')
        self.data_root = data_root
        self.spec = MelSpectrogram()
        self.mode = mode

        # self.trans = transforms.Compose([
        #     transforms.Lambda(lambda im: im.float() / 255.0),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        # 口唇動画、音声データまでのパス一覧を取得
        self.items = get_datasets(self.data_root, self.mode)
        self.len = len(self.items)
        # self.duration = duration

        print(f'Size of {type(self).__name__}: {self.len}')

        #random.shuffle(self.items)
        # イテレータを生成
        #self.item_iter = iter(self.items)

        # self.current_item = None
        # self.current_item_attributes = dict()

    def __len__(self):
        return self.len

    # def reset_item(self):
    #     print("# reset_item ... #")
    #     self.current_item = None
    #     return self['item']

    # def get_item(self):
    #     print("# get_item ... #")
    #     try:
    #         # イテレータの次の要素を受け取る
    #         item_idx = next(self.item_iter)
    #     except StopIteration:
    #         # nextでもうなかったときはitemsをシャッフルしてイテレータを作り直す
    #         random.shuffle(self.items)
    #         self.item_iter = iter(self.items)
    #         item_idx = next(self.item_iter)

    #     worker_info = get_worker_info()
    #     if worker_info:
    #         item_idx = (item_idx + worker_info.id) % len(self.items)
        
    #     video_path, audio_path = self.items[item_idx]

    #     try:
    #         video_info = ffmpeg.probe(video_path)["format"]
    #     except:
    #         return self.get_item()

    #     self.current_item = self.items[item_idx] 
    #     self.current_item_attributes = {
    #         'start_time': 0,
    #         'end_time': x_round(float(video_info['duration']))
    #     }
    #     return self.current_item

    # __getitem__()の挙動が見たい時は (self, _) -> (self)で一応見れます
    def __getitem__(self, index):
        #item_idx = next(self.item_iter)
        video_path, audio_path = self.items[index]
        print("\ngetting data...")
        print(f"video_path = {video_path}")
        print(f"audio_path = {audio_path}")
        data_path = Path(video_path)
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
            mode=None,
        )







        # overlap = 0.1
        # start_time = max(self.current_item_attributes['start_time'] - overlap, 0)
        # print(start_time)
        # end_time = self.current_item_attributes['end_time']

        # # start_timeからend_time-1の間で、0.5秒ずつ刻んだランダムなタイミングをstart_timeに設定
        # # 1秒間のデータを取りたいのでend_time-1しています
        # # get_timing_step = 0.5
        # # start_time = random.choice(np.arange(start_time, end_time-1, get_timing_step))

        # if start_time > end_time:
        #     return self.reset_item()

        # # データの時間を3つの選択肢からランダムに選ぶ
        # # duration = random.choice(np.arange(0.5, self.duration + overlap, overlap))
        # duration = self.duration    # 1秒
        # # 取った1秒分足しておく
        # self.current_item_attributes['start_time'] += duration

        

        return ret
        

def main():
    print("############################ Start!! ############################")
    datasets = KablabDataset(data_root, mode="train")
    breakpoint()
    # datasets = KablabDataset(data_root, mode="test")
    loader = DataLoader(
        dataset=datasets,
        batch_size=1,   
        shuffle=False,
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
            ret = batch    
            print("################################################")
            print(type(ret[0]))
            print(type(ret[1]))
            print(type(ret[2]))
            print(type(ret[3]))
            print(f"lip = {ret[0].shape}")
            print(f"y(acoustic features) = {ret[1].shape}")
            print(f"mask = {ret[2].shape}")
            print(f"feat_add = {ret[3].shape}")




if __name__ == "__main__":
    # print(data_root)
    main()
    