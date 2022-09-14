import hydra

from pathlib import Path
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import torch

from data_check import save_data
from train_ae_audio import make_model
from generate import make_test_loader, get_path
from calc_accuracy import calc_accuracy
from dataset.dataset_npz import get_datasets

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, test_loader, dataset, device, save_path, ref_loader):
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
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        # speaker embeddingのために使用する音響特徴量
        _, _, feature_ref, _, _, _, _, _ = ref_loader.next()
        feature_ref = feature_ref.to(device)

        start_time = time.time()

        with torch.no_grad():
            output, feat_add_out, phoneme, spk_emb, enc_output = model(lip=lip, feature_ref=feature_ref)

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

        iter_cnt += 1
        if iter_cnt == 53:
            break

    return process_times


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)
    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/ae/lip/2022:09:10_09-24-50/mspec80_100.ckpt") 
    
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

    data_root_list, mean_std_path, save_path_list = get_path(cfg, model_path)

    ref_data_root = Path(cfg.train.lip_pre_loaded_path_9696_time_only).expanduser()
    ref_mean_std_path = Path(cfg.train.lip_mean_std_path_9696_time_only).expanduser()
    ref_loader, ref_dataset = make_test_loader(cfg, ref_data_root, ref_mean_std_path)
    ref_loader = iter(ref_loader)

    # 同じ発話内容のものを使いたくないので適当に取り出しておく
    for _ in range(100):
        _ = ref_loader.next()

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

        print("--- generate ---")
        process_times = generate(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
            ref_loader=ref_loader,
        )

        print("--- calc accuracy ---")
        calc_accuracy(save_path, save_path.parents[0], cfg, process_times)
        

if __name__ == "__main__":
    main()