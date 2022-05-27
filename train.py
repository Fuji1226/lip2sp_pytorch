"""
reference
https://github.com/Rudrabha/Lip2Wav.git
"""

from pathlib import Path
import os

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset import av_speech_collate_fn_pad, x_round, KablabDataset
from hparams import create_hparams
from model.net import PreNet


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


def train(data_root, hparams):
    ###モデルにデータの入力、誤差の算出、逆伝搬によるパラメータの更新を行う####

    # Dataloader作成
    train_loader = make_train_loader(data_root, hparams, mode="train")
    test_loader = make_test_loader(data_root, hparams, mode="test")

    # model作成
    model = PreNet(in_channels=hparams.video_channels, out_channels=hparams.n_mel_channels)
    # print(model)

    # 最適化手法
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.lr, betas=hparams.betas
    )
    # print(optimizer)

    # 損失関数
    loss = nn.MSELoss()


    return


def main():
    ###ここにデータセットモデルのインスタンス作成train関数を回す#####
<<<<<<< HEAD
    print('test')
    print('branch test')
    return
=======
>>>>>>> minami

    # datasetディレクトリまでのパス
    data_root = Path(get_datasetroot()).expanduser()    # users/minami/dataset

    # パラメータ取得
    hparams = create_hparams()

    # training
    train(data_root, hparams)

    return


if __name__=='__main__':
    main()