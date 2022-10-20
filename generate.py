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

import hydra

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset.dataset_npz import KablabDataset, KablabTransform, get_datasets
from data_check import save_data
from train_default import make_model
from calc_accuracy import calc_accuracy
from utils import make_test_loader, get_path_test, make_train_val_loader

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, test_loader, dataset, device, save_path):
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

# def generate_loop(cfg, model, test_loader, dataset, device, save_path):
#     model.eval()

#     lip_mean = dataset.lip_mean.to(device)
#     lip_std = dataset.lip_std.to(device)
#     feat_mean = dataset.feat_mean.to(device)
#     feat_std = dataset.feat_std.to(device)
#     feat_add_mean = dataset.feat_add_mean.to(device)
#     feat_add_std = dataset.feat_add_std.to(device)

#     process_times = []

#     iter_cnt = 0
#     for batch in tqdm(test_loader, total=len(test_loader)):
#         wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
#         lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

#         mel_input = torch.zeros([1, 80, 1], dtype=feature.dype).to(device)
#         start_time = time.time()

#         breakpoint()
#         with torch.no_grad():
#             for i in range(feature.shape[-1]):

#             output, dec_output, feat_add_out = model(lip)
#             #tf_output, _, _ = model(lip=lip, prev=feature)

#         end_time = time.time()
#         process_time = end_time - start_time
#         process_times.append(process_time)

#         _save_path = save_path / label[0]

#         #_save_path_tf = save_path / label[0]+'_tf'
#         os.makedirs(_save_path, exist_ok=True)
       
#         save_data(
#             cfg=cfg,
#             save_path=_save_path,
#             wav=wav,
#             lip=lip,
#             feature=feature,
#             feat_add=feat_add,
#             output=output,
#             lip_mean=lip_mean,
#             lip_std=lip_std,
#             feat_mean=feat_mean,
#             feat_std=feat_std,
#         )

#         # save_data(
#         #     cfg=cfg,
#         #     save_path=_save_path_tf,
#         #     wav=wav,
#         #     lip=lip,
#         #     feature=feature,
#         #     feat_add=feat_add,
#         #     output=tf_output,
#         #     lip_mean=lip_mean,
#         #     lip_std=lip_std,
#         #     feat_mean=feat_mean,
#         #     feat_std=feat_std,
#         # )


#         iter_cnt += 1
#         if iter_cnt == 53:
#             break

#     return process_times

def generate_for_train_dataset(cfg, model,train_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in tqdm(train_loader, total=len(train_loader)):
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        print('lip shape')
        print(lip.shape)
        start_time = time.time()

        with torch.no_grad():
            output, dec_output, feat_add_out = model(lip)
            #tf_output, _, _ = model(lip=lip, prev=feature)

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


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)

    model_path = Path("/home/usr1/q70261a/lip2sp_pytorch_all/lip2sp_920_re/check_point/default/lip/transfomer_check/mspec80_300.ckpt")
    if cfg.model_path is not None:
        model_path = Path(cfg.model_path)

    print('model path')
    print(str(model_path))
    if model_path.suffix == ".ckpt":
        try:
            model.load_state_dict(torch.load(str(model_path))['model'])
        except:
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['model'])
    elif model_path.suffix == ".pth":
        try:
            model.load_state_dict(torch.load(str(model_path)))
        except:
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

    data_root_list, mean_std_path, save_path_list = get_path_test(cfg, model_path)
    
    test = True
    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)
        train_loader, _, train_dataset, _ = make_train_val_loader(cfg, data_root, mean_std_path)

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
        else:
            process_times = generate_for_train_dataset(
                cfg=cfg,
                model=model,
                test_loader=train_loader,
                dataset=train_dataset,
                device=device,
                save_path=save_path,
            )
            
        print("--- calc accuracy ---")
        calc_accuracy(save_path, save_path.parents[0], cfg, process_times)

if __name__ == "__main__":
    main()