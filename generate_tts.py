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
from data_check import save_data_tts
from train_tts import make_model
#from calc_accuracy import calc_accuracy
from utils import make_test_loader_tts, get_path_test_tts, make_train_val_loader_tts

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

    process_times = []

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
    #for batch in test_loader:
        print(f'iter cnt: {iter_cnt}')
        wav, feature, text, stop_token, feature_len, text_len, filename, label = batch
        
        
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            print(f'generate {iter_cnt}')
            dec_output, output, logit, att_w = model(text, text_len)
            #tf_output, _, _ = model(lip=lip, prev=feature)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / filename[0]

        #_save_path_tf = save_path / label[0]+'_tf'
        os.makedirs(_save_path, exist_ok=True)
       
        save_data_tts(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            feature=feature,
            output=output,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        
        check_attention_weight(att_w[0], "attention_FR", _save_path)
    return process_times


@hydra.main(config_name="config_tts", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)
    #model_path = Path("/home/usr1/q70261a/lip2sp_pytorch_all/lip2sp_920_re/check_point/default/lip/transfomer_check/mspec80_300.ckpt")
    if cfg.model_path is not None:
        model_path = Path(cfg.model_path)

    # path = '/home/naoaki/lip2sp_pytorch_all/lip2sp_920_re/check_point/tts/tts/2023:09:30_06-22-07_fujtia_last/mspec80_2.ckpt'
    # model_path = Path(path)
    
    print('model path')
    print(str(model_path))
    #breakpoint()
    if model_path.suffix == ".ckpt":
        try:
            model.load_state_dict(torch.load(str(model_path))['model'])
        except:
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['model'])
    elif model_path.suffix == ".pth":
        try:
            print('aaa')
            model.load_state_dict(fix_model_state_dict(torch.load(str(model_path))))
        except:
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))


    data_root_list, save_path_list, train_data_root = get_path_test_tts(cfg, model_path)
    
    print(f'save path: {save_path_list[0]}')
    test = True

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader_tts(cfg, data_root, train_data_root)

        print("--- generate ---")
        if test:
            process_times = generate(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )
            
        print("--- calc accuracy ---")
        #calc_accuracy(save_path, save_path.parents[0], cfg, process_times)

if __name__ == "__main__":
    main()