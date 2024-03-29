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

from data_check import save_data
from train_default_vq import make_model
from calc_accuracy import calc_accuracy
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature

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

    input_length = cfg.model.lip_min_frame
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep = gen_separate(lip, input_length, shift_frame)
        
        start_time = time.time()

        with torch.no_grad():
            if cfg.train.use_gc:
                output, dec_output, vq_loss = model(lip=lip_sep, gc=speaker)
            else:
                output, dec_output, vq_loss = model(lip=lip_sep)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / label[0]
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

        # iter_cnt += 1
        # if iter_cnt == 53:
        #     break

    return process_times


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)

    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/default_vq/lip_st/2022:10:27_14-27-14/mspec80_300.ckpt")

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

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

        process_times = None
        print("--- generate ---")
        process_times = generate(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )
        
    for data_root, save_path in zip(data_root_list, save_path_list):
        print("--- calc accuracy ---")
        calc_accuracy(save_path, save_path.parents[0], cfg, process_times)

if __name__ == "__main__":
    main()