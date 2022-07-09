"""
保存したパラメータを読み込み,合成を行う際に使用してください
まだ色々確認の段階で,printとかがたくさんあってとても汚くなってます…
一旦モデルのパラメータをロードし,音声合成を行うことは可能です

1. model_pathの変更
    main()の中にあります
    自分が読み込みたいやつまでのパスを通してください(checkpointやresult/train)

2. conf/testの変更
    顔と口唇のどちらのデータを使うか
    学習用データとテスト用データのどちらを使うか
    など

    顔か口唇かを学習時の状態に合わせれば基本的には大丈夫だと思います

3. generate.pyの実行

4. 結果の確認
    result/generateの下に保存されると思います
"""


from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow

import wandb

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_npz import KablabDataset, KablabTransform
from model.models import Lip2SP
from data_check import save_data
from train_wandb import make_model


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def make_test_loader(cfg, data_path, mean_std_path):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    dataset = KablabDataset(
        data_path=data_path,
        mean_std_path=mean_std_path,
        name=cfg.model.name,
        train=False,
        transform=trans,
        cfg=cfg,
        debug=cfg.test.debug
    )
    # 学習用データで確認するとき用
    # dataset = KablabDataset(
    #     data_root=cfg.train.pre_loaded_path,    # npzファイルまでのパス
    #     mean_std_path=cfg.train.mean_std_path,
    #     name=cfg.model.name,
    #     train=True,
    #     transform=trans,
    #     cfg=cfg,
    #     debug=cfg.test.debug
    # )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=1,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, dataset


def generate(cfg, model, test_loader, datasets, device, save_path):
    index = 0

    lip_mean = datasets.lip_mean.to(device)
    lip_std = datasets.lip_std.to(device)
    feat_mean = datasets.feat_mean.to(device)
    feat_std = datasets.feat_std.to(device)
    feat_add_mean = datasets.feat_add_mean.to(device)
    feat_add_std = datasets.feat_add_std.to(device)

    training_method = "ss"
    mixing_prob = 0.1
    epoch = 1
    iter_cnt = 1
    visualize = False

    inference_enc_list = []
    inference_dec_list = []
    forward_enc_list = []    
    forward_dec_list = []

    for batch in test_loader:
        model.eval()
        index += 1

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        with torch.no_grad():
            if model.which_decoder == "transformer":
                for_output, for_dec_output, for_enc_output = model(
                    lip=lip,
                    data_len=data_len,
                    prev=feature,
                    training_method=training_method,
                    mixing_prob=mixing_prob,
                    visualize=visualize,
                    epoch=epoch,
                    iter_cnt=iter_cnt,
                )              
                forward_enc_list.append(enc_output)
                forward_dec_list.append(dec_output)

                output, dec_output, enc_output = model(lip, visualize=False)
                inference_enc_list.append(enc_output)
                inference_dec_list.append(dec_output)

            # if cfg.test.generate_method == "inference":
            #     output, dec_output, enc_output = model(lip, visualize=False)

            # elif cfg.test.generate_method == "forward":
            #     output, dec_output, enc_output = model(
            #         lip=lip,
            #         data_len=data_len,
            #         prev=feature,
            #         training_method=training_method,
            #         mixing_prob=mixing_prob,
            #         visualize=visualize,
            #         epoch=epoch,
            #         iter_cnt=iter_cnt,
            #     )              

        # ディレクトリ作成
        input_save_path = os.path.join(save_path, label[0], 'input')
        output_save_path = os.path.join(save_path, label[0], 'output')
        os.makedirs(input_save_path, exist_ok=True)
        os.makedirs(output_save_path, exist_ok=True)
        
        save_data(
            cfg=cfg,
            input_save_path=input_save_path,
            output_save_path=output_save_path,
            index=index,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            output=output,
            dec_output=dec_output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        plt.close()
        if index > 0:
            break
    
    count = 0
    for i in range(len(inference_enc_list)):
        for j in range(inference_enc_list[i].shape[0]):
            for k in range(inference_enc_list[i].shape[1]):
                for l in range(inference_enc_list[i].shape[2]):
                    if inference_enc_list[i][j, k, l] == forward_enc_list[i][j, k, l]:
                        count += 1
    
    numel_count = 0
    for i in range(len(inference_enc_list)):
        numel_count += inference_enc_list[i].numel()
    print(f"\ninference_enc_list and forward_enc_list check")
    print(f"count = {count}")
    print(f"numel_count = {numel_count}")
    print(count == numel_count)


    count = 0
    for i in range(len(forward_enc_list) - 1):
        for j in range(forward_enc_list[i].shape[0]):
            for k in range(forward_enc_list[i].shape[1]):
                for l in range(forward_enc_list[i].shape[2] - 100):
                    if forward_enc_list[i][j, k, l] == forward_enc_list[i+1][j, k, l]:
                        count += 1
    print(f"\nforward_enc_list check")
    print(f"count = {count}")
    print(f"numel_count = {numel_count}")


    count = 0
    for i in range(len(inference_dec_list) - 1):
        for j in range(inference_dec_list[i].shape[0]):
            for k in range(inference_dec_list[i].shape[1]):
                for l in range(inference_dec_list[i].shape[2] - 100):
                    if inference_dec_list[i][j, k, l] == inference_dec_list[i+1][j, k, l]:
                        count += 1
    
    numel_count = 0
    for i in range(len(inference_enc_list)):
        numel_count += inference_enc_list[i].numel()
    print("\ninference dec_output check")
    print(f"count = {count}")
    print(f"numel_count = {numel_count}")

    count = 0
    for i in range(len(inference_dec_list) - 1):
        for j in range(inference_dec_list[i].shape[0]):
            for k in range(inference_dec_list[i].shape[1]):
                if inference_dec_list[i][j, k, 0] == forward_dec_list[i][j, k, 0]:
                    count += 1
    print("\ndec_output first frame check")
    print(f"count = {count}")

    count_inf = 0
    count_for = 0
    for i in range(len(inference_dec_list) - 1):
        for j in range(inference_dec_list[i].shape[0]):
            for k in range(inference_dec_list[i].shape[1]):
                if inference_dec_list[i][j, k, 0] == inference_dec_list[i+1][j, k, 0]:
                    count_inf += 1
                if forward_dec_list[i][j, k, 0] == forward_dec_list[i+1][j, k, 0]:
                    count_for += 1
    print("\ndec_output first frame comparison")
    print(f"count_inf = {count_inf}")
    print(f"count_for = {count_for}")


    print("first frame check(最初はどちらも0入力なので，同じになるはず…)")
    # dec_output_count = 0
    # prenet_count = 0
    # dec_self_atten_count = 0
    # dec_enc_atten_count = 0
    # for i in range(len(inference_dec_list)):
    #     for j in range(inference_dec_list[i].shape[0]):
    #         for k in range(inference_dec_list[i].shape[1]):


    



@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # 口唇動画か顔かの選択
    lip_or_face = cfg.test.face_or_lip
    assert lip_or_face == "face" or "lip"
    if cfg.test.which_data == "test":
        if lip_or_face == "face":
            data_path = cfg.test.face_pre_loaded_path
            mean_std_path = cfg.test.face_mean_std_path
        elif lip_or_face == "lip":
            data_path = cfg.test.lip_pre_loaded_path
            mean_std_path = cfg.test.lip_mean_std_path

    elif cfg.test.which_data == "train":
        if lip_or_face == "face":
            data_path = cfg.train.face_pre_loaded_path
            mean_std_path = cfg.train.face_mean_std_path
        elif lip_or_face == "lip":
            data_path = cfg.train.lip_pre_loaded_path
            mean_std_path = cfg.train.lip_mean_std_path

    print("--- data directory check ---")
    print(f"data_path = {data_path}")
    print(f"mean_std_path = {mean_std_path}")

    #インスタンス作成
    model = make_model(cfg, device)

    # パラメータのロード
    # model_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/face/2022:07:08_20-10-06/mspec_30.ckpt"     # glu
    model_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/face/2022:07:07_00-58-32/mspec_80.ckpt"     # transformer
    model.load_state_dict(torch.load(model_path)['model'])

    # model_path = "/home/usr4/r70264c/lip2sp_pytorch/result/train/2022:06:23_10-07-13/model_mspec.pth"
    # model.load_state_dict(torch.load(model_path))

    # 保存先
    save_path = cfg.test.generate_save_path
    # save_path = os.path.join(save_path, Path(cfg.test.model_save_path).name)
    save_path = os.path.join(save_path, lip_or_face, Path(model_path).parents[0].name, Path(model_path).stem, cfg.test.which_data, cfg.test.generate_method)
    os.makedirs(save_path, exist_ok=True)

    # Dataloader作成
    test_loader, datasets = make_test_loader(cfg, data_path, mean_std_path)

    # generate
    model.eval()
    generate(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        datasets=datasets,
        device=device,
        save_path=save_path,
    )
    

if __name__ == "__main__":
    main()
    # test()