"""
user/minami/dataset/lip/lip_cropped
このディレクトリに口唇部分を切り取った動画と、wavデータを入れておけば動くと思います!
"""

from doctest import UnexpectedException
from pathlib import Path
import os
import time
from datetime import datetime
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
from loss import masked_mse, ls_loss, fm_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator


current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# パラメータの保存
def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


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


def train_one_epoch(model: nn.Module, discriminator, data_loader, optimizer, loss_f, device, hparams):
    epoch_loss = 0
    data_cnt = 0
    for batch in data_loader:
        (lip, target, feat_add), data_len = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        ################順伝搬###############
        # output : postnet後の出力
        # dec_output : postnet前の出力
        output, dec_output = model(
            lip=lip,
            data_len=data_len,
            prev=target,
        )                      
        ####################################
        
        # GAN試し
        if hparams.which_d is not None:
            out_f, fmaps_f = discriminator(output[:, :, :-2])
            out_r, fmaps_r = discriminator(target[:, :, 2:])
            loss_d = ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=hparams.which_d, which_loss="d")
            loss_g_ls = ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=hparams.which_d, which_loss="g")
            loss_g_fm = fm_loss(fmaps_f, fmaps_r, data_len, max_len=model.max_len * 2, which_d=hparams.which_d)
            print(loss_d)
            print(loss_g_ls)
            print(loss_g_fm)
        
        loss_default = loss_f(output[:, :, :-2], target[:, :, 2:]) + loss_f(dec_output[:, :, :-2], target[:, :, 2:])
        # パディング部分をマスクした損失を計算。postnet前後の出力両方を考慮。
        loss = masked_mse(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + masked_mse(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 

        print(loss_default)
        print(loss)

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
        n_position=hparams.length,  # 口唇動画に対して長ければいい
        max_len=hparams.length // 2,
        which_decoder=hparams.which_decoder,
        training_method=hparams.training_method,
        num_passes=hparams.num_passes,
        mixing_prob=hparams.mixing_prob,
        dropout=hparams.dropout,
        reduction_factor=hparams.reduction_factor,
        use_gc=hparams.use_gc,
    )

    # discriminator
    if hparams.which_d is not None:
        if hparams.which_d == "unet":
            discriminator = UNetDiscriminator()
        elif hparams.which_d == "jcu":
            discriminator = JCUDiscriminator(
                in_channels=hparams.out_channels,
                out_channels=1,
                use_gc=hparams.use_gc,
                emb_in=hparams.batch_size,
            )
    else:
        discriminator = None
    
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.lr, betas=hparams.betas
    )

    # Dataloader作成
    train_loader = make_train_loader(data_root, hparams, mode="train")
    test_loader = make_test_loader(data_root, hparams, mode="test")

    # 損失関数
    loss_f = nn.MSELoss()
    train_loss_list = []

    # training
    for epoch in range(hparams.max_epoch):
        print(f"##### {epoch} #####")
        epoch_loss = train_one_epoch(model, discriminator, train_loader, optimizer, loss_f, device, hparams)
        train_loss_list.append(epoch_loss)
        print(f"epoch_loss = {epoch_loss}")
        print(f"train_loss_list = {train_loss_list}")
        
    save_result(train_loss_list, result_path+'/train_loss.png')


if __name__=='__main__':
    main()