"""
reference
https://github.com/Chris10M/Lip2Speech.git
"""

import os
import sys
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

from data_process.audio_process import MelSpectrogram
from get_dir import get_datasetroot, get_data_directory


data_root = Path(get_datasetroot()).expanduser()


def av_speech_collate_fn_pad(batch):
    lower_faces, speeches, melspecs = zip(*batch)
    
    max_frames_in_batch = max([l.shape[0] for l in lower_faces])
    max_samples_in_batch = max([s.shape[1] for s in speeches])
    max_melspec_samples_in_batch = max([m.shape[1] for m in melspecs])

    padded_lower_faces = torch.zeros(len(lower_faces), max_frames_in_batch, *tuple(lower_faces[0].shape[1:]))
    padded_speeches = torch.zeros(len(speeches), 1, max_samples_in_batch)
    padded_melspecs = torch.zeros(len(melspecs), melspecs[0].shape[0], max_melspec_samples_in_batch)
    mel_gate_padded = torch.zeros(len(melspecs), max_melspec_samples_in_batch)

    video_lengths = list()
    audio_lengths = list()
    melspec_lengths = list()
    for idx, (lower_face, speech, melspec) in enumerate(zip(lower_faces, speeches, melspecs)):
        T = lower_face.shape[0]
        video_lengths.append(T)

        padded_lower_faces[idx, :T, :, :, :] = lower_face

        S = speech.shape[-1]
        audio_lengths.append(S)
        padded_speeches[idx, :, :S] = speech
        
        M = melspec.shape[-1]
        melspec_lengths.append(M)
        padded_melspecs[idx, :, :M] = melspec

        mel_gate_padded[idx, M-1:] = 1.0

    padded_lower_faces = padded_lower_faces.permute(0, 2, 1, 3, 4)
    padded_speeches = padded_speeches.squeeze(1) 

    video_lengths = torch.tensor(video_lengths)
    audio_lengths = torch.tensor(audio_lengths)
    melspec_lengths = torch.tensor(melspec_lengths)

    return (padded_lower_faces, video_lengths), (padded_speeches, audio_lengths), (padded_melspecs, melspec_lengths, mel_gate_padded)


def x_round(x):
    return math.floor(x * 4) / 4


class KablabDataset(Dataset):
    def __init__(self, data_root, mode='train', duration=1):
        super().__init__()
        assert mode in ('train', 'test')
        self.data_root = data_root
        self.spec = MelSpectrogram()
        self.mode = mode

        self.trans = transforms.Compose([
            transforms.Lambda(lambda im: im.float() / 255.0),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.items = dict()
        idx = 0
        for curDir, Dir, Files in os.walk(self.data_root):
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
                            self.items[idx] = [video_path, audio_path]
                            idx += 1
                else:
                    continue
        self.len = len(self.items)
        self.duration = duration

        print(f'Size of {type(self).__name__}: {self.len}')

        random.shuffle(self.items)
        # イテレータを生成
        self.item_iter = iter(self.items)

        self.current_item = None
        self.current_item_attributes = dict()

    def __len__(self):
        return self.len

    def reset_item(self):
        self.current_item = None
        return self['item']

    def get_item(self):
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

    def __getitem__(self, _):
        if self.current_item is None:
            item = self.get_item()
        else:
            item = self.current_item

        # 動画、音声ファイルへのpath
        video_path, audio_path = item

        overlap = 0.2
        start_time = max(self.current_item_attributes['start_time'] - overlap, 0)
        end_time = self.current_item_attributes['end_time']

        if start_time > end_time:
            return self.reset_item()

        duration = random.choice(np.arange(0.5, self.duration + overlap, overlap))
        self.current_item_attributes['start_time'] += duration

        try:
            # speech : (C, T)
            speech, sampling_rate = torchaudio.load(audio_path, frame_offset=int(16000 * start_time), 
                                                               num_frames=int(16000 * duration), normalize=True, format='wav')                                    
        except:
            # traceback.print_exc()
            return self.reset_item()
        
        assert sampling_rate == 16000
        
        if speech.shape[1] == 0:
            return self.reset_item()

        # 返り値はvideo frames, audio frames, meta data。video framesしか使わない
        # frames : (T, H, W, C)
        frames, _, _ = torchvision.io.read_video(video_path, start_pts=start_time, end_pts=start_time + duration, pts_unit='sec')
        # (T, H, W, C) -> (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)

        # 正規化
        # frames = self.trans(frames)

        if frames.shape[0] == 0:
            return self.reset_item()

        try:
            melspec = self.spec(speech).squeeze(0)
        except:
            return self.reset_item()

        return frames, speech, melspec
        


def main():
    datasets = KablabDataset(data_root)
    loader = DataLoader(
        dataset=datasets,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=av_speech_collate_fn_pad
    )

    # print(datasets.__len__())
    # print(datasets.current_item)
    # # print(datasets.reset_item())
    # frames, speech, melspec = datasets.__getitem__()
    # print(f"type(faces) = {type(frames)}")
    # print(f"type(speech) = {type(speech)}")
    # print(f"type(melspec) = {type(melspec)}")


    # results
    for bdx, batch in enumerate(loader):
        (video, video_lengths), (speeches, audio_lengths), (melspecs, melspec_lengths, mel_gates) = batch
        
        frames = video
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
    