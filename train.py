"""
reference
https://github.com/joannahong/Lip2Wav-pytorch.git
https://github.com/Chris10M/Lip2Speech.git
"""

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset import av_speech_collate_fn_pad, x_round, KablabDataset
from hparams import create_hparams
from model.net import PreNet


current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def make_train_loader(data_root, hparams, mode):
    assert mode == "train"
    datasets = KablabDataset(data_root, mode)
    train_loader = DataLoader(
        dataset=datasets,
        batch_size=hparams.batch_size,   
        shuffle=False,
        num_workers=hparams.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=av_speech_collate_fn_pad
    )
    return train_loader

def make_test_loader(data_root, hparams, mode):
    assert mode == "test"
    datasets = KablabDataset(data_root, mode)
    test_loader = DataLoader(
        dataset=datasets,
        batch_size=hparams.batch_size,   
        shuffle=False,
        num_workers=hparams.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=av_speech_collate_fn_pad
    )
    return test_loader


# パラメータの保存
def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def train(data_root, hparams, device):
    ###モデルにデータの入力、誤差の算出、逆伝搬によるパラメータの更新を行う####

    # model作成
    model = PreNet(in_channels=hparams.video_channels, out_channels=hparams.n_mel_channels)

    # 最適化手法
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.lr, betas=hparams.betas
    )

    # load checkpoint
    # 保存したパラメータがすでにあるときにそれを読み込む
    iteration = 1

    # Dataloader作成
    train_loader = make_train_loader(data_root, hparams, mode="train")
    test_loader = make_test_loader(data_root, hparams, mode="test")


    # 損失関数
    loss = nn.MSELoss()


    model.train()
    print("================ MAIN TRAINNIG LOOP! ===================")
    while iteration <= hparams.max_iter:
        for batch in train_loader:
            if iteration > hparams.max_iter:
                break
            start = time.perf_counter()
            print(f"start = {start}")

            (videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates) = batch
            videos, audios, melspecs = videos.to(device), audios.to(device), melspecs.to(device)
            video_lengths, audio_lengths, melspec_lengths = video_lengths.to(device), audio_lengths.to(device), melspec_lengths.to(device)
            mel_gates = mel_gates.to(device)

            iteration += 1


    return


def main():
    ###ここにデータセットモデルのインスタンス作成train関数を回す#####
    print('test')
    print('branch test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # datasetディレクトリまでのパス
    data_root = Path(get_datasetroot()).expanduser()    # users/minami/dataset

    # パラメータ取得
    hparams = create_hparams()

    # training
    train(data_root, hparams, device)

    return


if __name__=='__main__':
    main()