"""
reference
https://github.com/Chris10M/Lip2Speech.git
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


def av_speech_collate_fn_pad(batch):
    """
    各バッチで口唇動画、音声サンプル、メルスペクトログラムをそれぞれ最大フレームに合わせるpaddingを行なっています

    lips : (T, C, H, W)
    speeches : (C=1, T)
    melspecs : (n_mel_channels, mel_frames)

    上だとpadded_lipが5次元にならないので、こっちかもしれないです
    lips : (B, T, C, H, W)
    speeches : (B, C=1, T)
    melspecs : (B, n_mel_channels, mel_frames)

    <追記>
    やっぱり下のデータ形状であってる気がします
    """
    print("# av_speech_collate_fn_pad ... #")
    lips, waveforms, melspecs = zip(*batch)

    # 口唇動画、音声サンプル、メルスペクトログラムそれぞれについて、バッチ内での最大フレーム数を計算
    max_frames_in_batch = max([l.shape[0] for l in lips])   # 50
    max_samples_in_batch = max([s.shape[1] for s in waveforms]) # 16000
    max_melspec_samples_in_batch = max([m.shape[1] for m in melspecs])  # 100
    # print(max_frames_in_batch)
    # print(max_samples_in_batch)
    # print(max_melspec_samples_in_batch)

    # バッチ内で最大のフレーム数のサンプルのやつに合わせるための0行列を作成
    # len(data) == data.shape[0]
    padded_lips = torch.zeros(len(lips), max_frames_in_batch, *tuple(lips[0].shape[1:]))  
    padded_waveforms = torch.zeros(len(waveforms), 1, max_samples_in_batch)
    padded_melspecs = torch.zeros(len(melspecs), melspecs[0].shape[0], max_melspec_samples_in_batch)
    mel_gate_padded = torch.zeros(len(melspecs), max_melspec_samples_in_batch)

    video_lengths = list()
    audio_lengths = list()
    melspec_lengths = list()
    # 0で初期化された行列に入れていくことで最大フレームに合わせている
    for idx, (lip, waveform, melspec) in enumerate(zip(lips, waveforms, melspecs)):
        T = lip.shape[0]
        video_lengths.append(T)

        padded_lips[idx, :T, :, :, :] = lip

        S = waveform.shape[-1]
        audio_lengths.append(S)
        padded_waveforms[idx, :, :S] = waveform
        
        M = melspec.shape[-1]
        melspec_lengths.append(M)
        padded_melspecs[idx, :, :M] = melspec

        mel_gate_padded[idx, M-1:] = 1.0

    # (len(lips), max_frames_in_batch, C, H, W) -> (len(lips), C, max_frames_in_batch, H, W)
    padded_lips = padded_lips.permute(0, 2, 1, 3, 4)
    # (len(waveform), 1, max_samples_in_batch) -> (len(waveform), max_samples_in_batch)
    padded_waveforms = padded_waveforms.squeeze(1) 

    video_lengths = torch.tensor(video_lengths)
    audio_lengths = torch.tensor(audio_lengths)
    melspec_lengths = torch.tensor(melspec_lengths)

    return (padded_lips, video_lengths), (padded_waveforms, audio_lengths), (padded_melspecs, melspec_lengths, mel_gate_padded)


def x_round(x):
    return math.floor(x * 4) / 4


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


class KablabDataset(Dataset):
    def __init__(self, data_root, mode='train', duration=1 - (1/hparams.fps)):
        super().__init__()
        assert mode in ('train', 'test')
        self.data_root = data_root
        # self.spec = MelSpectrogram()
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

        random.shuffle(self.items)
        # イテレータを生成
        self.item_iter = iter(self.items)

        self.current_item = None
        self.current_item_attributes = dict()

    def __len__(self):
        return self.len

    def reset_item(self):
        print("# reset_item ... #")
        self.current_item = None
        return self['item']

    def get_item(self):
        print("# get_item ... #")
        try:
            # イテレータの次の要素を受け取る
            item_idx = next(self.item_iter)
        except StopIteration:
            # nextでもうなかったときはitemsをシャッフルしてイテレータを作り直す
            random.shuffle(self.items)
            self.item_iter = iter(self.items)
            item_idx = next(self.item_iter)

        worker_info = get_worker_info()
        if worker_info:
            item_idx = (item_idx + worker_info.id) % len(self.items)
        
        video_path, audio_path = self.items[item_idx]

        try:
            video_info = ffmpeg.probe(video_path)["format"]
        except:
            return self.get_item()

        self.current_item = self.items[item_idx] 
        self.current_item_attributes = {
            'start_time': 0,
            'end_time': x_round(float(video_info['duration']))
        }
        return self.current_item

    # __getitem__()の挙動が見たい時は (self, _) -> (self)で一応見れます
    def __getitem__(self, _):
        print("# getting item... #")
        if self.current_item is None:
            item = self.get_item()
        else:
            item = self.current_item

        # 動画、音声ファイルへのpath
        video_path, audio_path = item
        print(video_path)
        print(audio_path)

        overlap = 0.1
        start_time = max(self.current_item_attributes['start_time'] - overlap, 0)
        print(start_time)
        end_time = self.current_item_attributes['end_time']

        # start_timeからend_time-1の間で、0.5秒ずつ刻んだランダムなタイミングをstart_timeに設定
        # 1秒間のデータを取りたいのでend_time-1しています
        # get_timing_step = 0.5
        # start_time = random.choice(np.arange(start_time, end_time-1, get_timing_step))

        if start_time > end_time:
            return self.reset_item()

        # データの時間を3つの選択肢からランダムに選ぶ
        # duration = random.choice(np.arange(0.5, self.duration + overlap, overlap))
        duration = self.duration    # 1秒
        # 取った1秒分足しておく
        self.current_item_attributes['start_time'] += duration

        #######################################
        # audio_process #
        #######################################
        try:
            # waveform : (C=1, T=16000)
            waveform, sampling_rate = torchaudio.load(audio_path, frame_offset=int(16000 * start_time), 
                                                               num_frames=int(16000 * duration), normalize=True, format='wav')                                    
        except:
            # traceback.print_exc()
            return self.reset_item()
        
        if waveform.shape[1] == 0:
            return self.reset_item()

        assert sampling_rate == 16000
        # assert waveform.shape[1] == 16000

        try:
            # melspec : (n_mel_channels, mel_frames)
            melspec = self.spec(waveform).squeeze(0)
        except:
            return self.reset_item()

        # assert melspec.shape[1] == hparams.fps * 2
        #######################################
        # video_process #
        #######################################

        # 動画のfpsが50か確認
        check_fps = ffmpeg.probe(video_path)['streams'][0]['avg_frame_rate']
        assert check_fps == "50/1"

        # 返り値はvideo frames, audio frames, meta data。video framesしか使わない
        # frames : (T, H, W, C)
        try:
            frames, _, _ = torchvision.io.read_video(video_path, start_pts=start_time, end_pts=start_time + duration, pts_unit='sec')
        except:
            return self.reset_item()

        if frames.shape[0] == 0:
            return self.reset_item()
        # print(frames.shape)
        # assert frames.shape[0] == 50
        # (T, H, W, C) -> (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)

        # frames = self.trans(frames)

        return frames, waveform, melspec
        


def main():
    datasets = KablabDataset(data_root, mode="train")
    # datasets = KablabDataset(data_root, mode="test")
    loader = DataLoader(
        dataset=datasets,
        batch_size=5,   
        shuffle=False,
        num_workers=0,      
        pin_memory=False,
        drop_last=True,
        collate_fn=av_speech_collate_fn_pad
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
    for bdx, batch in enumerate(loader):
        (video, video_lengths), (speeches, audio_lengths), (melspecs, melspec_lengths, mel_gates) = batch
        print("################################################")
        print("<video_data>")
        print(f"video.shape = {video.shape}")
        print(f"video_lengths.shape = {video_lengths.shape}, video_lengths = {video_lengths}")
        print("\n<audio_data>")
        print(f"speeches.shape = {speeches.shape}")
        print(f"audio_lengths.shape = {audio_lengths.shape}, audio_lengths = {audio_lengths}")
        print(f"melspecs.shape = {melspecs.shape}")
        print(f"melspec_lengths.shape = {melspec_lengths.shape}, melspec_lengths = {melspec_lengths}")
        print(f"mel_gates.shape = {mel_gates.shape}")




if __name__ == "__main__":
    # print(data_root)
    main()
    