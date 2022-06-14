"""
lip2sp_pytorch/conf/modelにあるyamlファイルのパスや、モデルのパラメータの読み込み先のパスを設定してから実行してください
"""


from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow

# import wandb
# wandb.init(
#     project='llip2sp_pytorch',
#     name="desk-test"
# )

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_no_chainer import KablabDataset, KablabTransform
from model.models import Lip2SP
from loss import masked_mse, delta_loss, ls_loss, fm_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator
from train import make_test_loader, make_train_loader
from data_process.feature import mel2wave
from data_check import save_data


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def generate(cfg, model, test_loader, datasets, device, save_path):
    outputs = []
    dec_outputs = []
    index = 0

    lip_mean = datasets.lip_mean.to(device)
    lip_std = datasets.lip_std.to(device)
    feat_mean = datasets.feat_mean.to(device)
    feat_std = datasets.feat_std.to(device)
    feat_add_mean = datasets.feat_add_mean.to(device)
    feat_add_std = datasets.feat_add_std.to(device)

    for batch in test_loader:
        model.eval()

        (lip, target, feat_add), data_len, label = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)
        
        with torch.no_grad():
            output, dec_output = model.inference(
                lip=lip
            )
        outputs.append(output)
        dec_outputs.append(dec_output)

        # ディレクトリ作成
        try:
            os.makedirs(f"{save_path}/{label[0]}/input")
            os.makedirs(f"{save_path}/{label[0]}/output")
        except FileExistsError:
            pass
        input_save_path = f"{save_path}/{label[0]}/input"
        output_save_path = f"{save_path}/{label[0]}/output"
        
        save_data(
            cfg=cfg,
            input_save_path=input_save_path,
            output_save_path=output_save_path,
            index=index,
            lip=lip,
            feature=target,
            feat_add=feat_add,
            output=output,
            dec_output=dec_output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        index += 1

    return outputs, dec_outputs


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # 保存先
    save_path = cfg.model.generate_save_path
    try:
        os.makedirs(f"{save_path}/{current_time}")
    except FileExistsError:
        pass
    save_path = os.path.join(save_path, current_time)
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    #インスタンス作成
    model = Lip2SP(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        glu_inner_channels=cfg.model.d_model,
        glu_layers=cfg.model.glu_layers,
        pre_in_channels=cfg.model.pre_in_channels,
        pre_inner_channels=cfg.model.pre_inner_channels,
        post_inner_channels=cfg.model.post_inner_channels,
        n_position=cfg.model.length * 5,
        max_len=cfg.model.length // 2,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        training_method=cfg.train.training_method,
        num_passes=cfg.train.num_passes,
        mixing_prob=cfg.train.mixing_prob,
        dropout=cfg.train.dropout,
        reduction_factor=cfg.model.reduction_factor,
        use_gc=cfg.train.use_gc
    )
    model = model.to(device)

    # 保存したパラメータの読み込み
    # model_path = cfg.model.train_save_path+f'/2022:06:09_23-13-18/model_{cfg.model.name}.pth'

    # mlflowを利用したモデルの読み込み
    model_path = cfg.model.train_save_path + "/state_dict.pth"
    model.load_state_dict(torch.load(model_path))

    # Dataloader作成
    test_loader, datasets = make_test_loader(cfg)

    # generate
    model.eval()
    output, dec_output = generate(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        datasets=datasets,
        device=device,
        save_path=save_path,
    )
    

@hydra.main(config_name="config", config_path="conf")
def test(cfg):
    model_uri = 'runs:/665f72b343a34098ad93e82c504b7e03/model'
    model_path_dir = "model"
    run_id = "665f72b343a34098ad93e82c504b7e03"
    loaded_model = mlflow.pytorch.load_model(model_uri)

    return


if __name__ == "__main__":
    main()
    # test()