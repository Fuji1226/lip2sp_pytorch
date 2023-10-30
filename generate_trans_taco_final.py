"""
保存したパラメータを読み込み,合成を行う際に使用してください

1. model_pathの変更
    main()の中にあります
    自分が読み込みたいやつまでのパスを通してください(checkpointやresult/train)

2. conf/testの変更
    顔か口唇かを学習時の状態に合わせれば基本的には大丈夫だと思います

3. generate.pyの実行

4. 結果の確認
    result/generateの下に保存されると思います
"""

import matplotlib.pyplot as plt
from librosa.display import specshow

from collections import OrderedDict

import hydra

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from dataset.dataset_npz import KablabDataset, KablabTransform, get_datasets
from data_check import save_data_final
from train_taco_all_lip_from_tts import make_model
from calc_accuracy import calc_accuracy
from utils import make_test_loader_final, get_path_train

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def check_attention_weight(att_w, filename, save_path):
    att_w = att_w.to('cpu').detach().numpy().copy()

    plt.figure()
    sns.heatmap(att_w, cmap="viridis", cbar=True)
    plt.title("attention weight")
    plt.xlabel("text")
    plt.ylabel("feature")

    att_path = save_path / f"{filename}.png"
    plt.savefig(str(att_path))
    plt.close()

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    #breakpoint()
    return new_state_dict


def generate(cfg, model, test_loader, dataset, device, save_path):
    print('start genearete')
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        print(f'iter cnt: {iter_cnt}')
        lip = batch['lip'].to(device)
        data_len = batch['data_len'].to(device)
        label = batch['label']
    
        
        start_time = time.time()
        with torch.no_grad():
            print(f'generate {iter_cnt}')
            all_output = model(lip, data_len=data_len)
            #tf_output, _, _ = model(lip=lip, prev=feature)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        os.makedirs(_save_path, exist_ok=True)
       
        print(f'save path: {_save_path}')
        save_data_final(
            cfg=cfg,
            save_path=_save_path,
            batch=batch,
            all_output=all_output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        
        check_attention_weight(all_output['att_w'][0], "attention_FR", _save_path)


        iter_cnt += 1
        if iter_cnt == 53:
            break

    return process_times


def generate_for_train_check(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        start_time = time.time()
        with torch.no_grad():
            output, dec_output, feat_add_out = model(lip)
            #tf_output, _, _ = model(lip=lip, prev=feature)

        breakpoint()
        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / label[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        os.makedirs(_save_path, exist_ok=True)
       
        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        # save_data(
        #     cfg=cfg,
        #     save_path=_save_path_tf,
        #     wav=wav,
        #     lip=lip,
        #     feature=feature,
        #     feat_add=feat_add,
        #     output=tf_output,
        #     lip_mean=lip_mean,
        #     lip_std=lip_std,
        #     feat_mean=feat_mean,
        #     feat_std=feat_std,
        # )


        iter_cnt += 1
        if iter_cnt == 53:
            break

    return process_times




@hydra.main(config_name="config_all_from_tts_desk", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)
    
    # path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/test/mspec80_330.ckpt'
    # model_path = Path(path)
    
    
    if cfg.model_path is not None:
        model_path = Path(cfg.model_path)

    
    # print('model path')
    # print(str(model_path))

    # path
    _, mean_std_path, ckpt_path, _, _ = get_path_train(cfg, current_time)

    
    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem
    
    test_data_root = [Path(cfg.test.face_pre_loaded_path).expanduser()]
    print(f'test root: {test_data_root}')

    test_loader, test_dataset = make_test_loader_final(cfg, test_data_root, mean_std_path)
    
    
    print("--- generate ---")
    
    generate(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        dataset=test_dataset,
        device=device,
        save_path=save_path,
    )
            
    # print("--- calc accuracy ---")
    # calc_accuracy(save_path, save_path.parents[0], cfg, process_times)

if __name__ == "__main__":
    main()