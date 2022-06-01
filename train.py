"""
user/minami/dataset/lip/lip_cropped
このディレクトリに口唇部分を切り取った動画と、wavデータを入れておけば動くと思います!
"""

from genericpath import exists
from pathlib import Path
import os
import time
from datetime import datetime
from urllib.request import HTTPPasswordMgrWithDefaultRealm
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_remake import KablabDataset
from hparams import create_hparams
from model.models import Lip2SP

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# パラメータの保存
def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)

"""
def train(data_root, hparams, device):
    ###モデルにデータの入力、誤差の算出、逆伝搬によるパラメータの更新を行う####

    # model作成

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
    breakpoint()

    loss_list = []
    model.train()
    print("================ MAIN TRAINNIG LOOP! ===================")
    # for epoch in range(hparams.max_epoch):
    #     loss = 
    # while iteration <= hparams.max_iter:
    #     for batch in train_loader:
    #         if iteration > hparams.max_iter:
    #             break
    #         start = time.perf_counter()
    #         print(f"start = {start}")

    #         (videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates) = batch
    #         videos, audios, melspecs = videos.to(device), audios.to(device), melspecs.to(device)
    #         video_lengths, audio_lengths, melspec_lengths = video_lengths.to(device), audio_lengths.to(device), melspec_lengths.to(device)
    #         mel_gates = mel_gates.to(device)

    #         iteration += 1
    # return

"""


def make_train_loader(data_root, hparams, mode):
    assert mode == "train"
    datasets = KablabDataset(data_root, mode)
    train_loader = DataLoader(
        dataset=datasets,
        batch_size=hparams.batch_size,   
        shuffle=True,
        num_workers=hparams.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )
    return train_loader


def make_test_loader(data_root, hparams, mode):
    assert mode == "test"
    datasets = KablabDataset(data_root, mode)
    test_loader = DataLoader(
        dataset=datasets,
        batch_size=hparams.batch_size,   
        shuffle=True,
        num_workers=hparams.num_workers,      
        pin_memory=False,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader


def train_one_epoch(model: nn.Module, data_loader, optimizer, loss_f, device):
    epoch_loss = 0
    data_cnt = 0
    for batch in data_loader:
        (lip, target, feat_add), data_len = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        ################順伝搬###############
        output = model(
            lip=lip,
            data_len=data_len,
            prev=target,
            which_decoder="transformer",
        )                        ##modelの入力はおいおい
        ####################################

        loss = loss_f(output, target)           #targetはおいおい
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= data_cnt
    return epoch_loss

def save_result(loss_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(save_path)


def main():
    ###ここにデータセットモデルのインスタンス作成train関数を回す#####

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # datasetディレクトリまでのパス
    data_root = Path(get_datasetroot()).expanduser()    # users/minami/dataset
    # パラメータ取得
    hparams = create_hparams()

    #resultの表示
    result_path = 'results'
    os.makedirs(result_path, exist_ok=True)


    #インスタンス作成
    model = Lip2SP(
        in_channels=5, 
        out_channels=hparams.out_channels,
        res_layers=hparams.res_layers,
        d_model=hparams.d_model,
        n_layers=hparams.n_layers,
        n_head=hparams.n_head,
        d_k=hparams.d_k,
        d_v=hparams.d_v,
        d_inner=hparams.d_inner,
        glu_inner_channels=hparams.glu_inner_channels,
        glu_layers=hparams.glu_layers,
        pre_in_channels=hparams.pre_in_channels,
        pre_inner_channels=hparams.pre_inner_channels,
        post_inner_channels=hparams.post_inner_channels,
        dropout=hparams.dropout,
        n_position=hparams.length // 2,
        reduction_factor=hparams.reduction_factor,
        use_gc=hparams.use_gc,
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.lr, betas=hparams.betas
    )

    # Dataloader作成
    train_loader = make_train_loader(data_root, hparams, mode="train")
    test_loader = make_test_loader(data_root, hparams, mode="test")

    loss_f = nn.MSELoss()
    train_loss_list = []

    # training
    for epoch in range(hparams.max_epoch):
        print(f"##### {epoch} #####")
        epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_f, device)
        train_loss_list.append(epoch_loss)
        print(f"epoch_loss = {epoch_loss}")
        print(f"train_loss_list = {train_loss_list}")
        
    save_result(train_loss_list, result_path+'/train_loss.png')


if __name__=='__main__':
    main()