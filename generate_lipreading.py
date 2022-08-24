from unicodedata import name
from omegaconf import DictConfig, OmegaConf
import hydra

from pathlib import Path
import os
from pathlib import Path
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
from dataset.dataset_lipreading import LipReadingDataset, LipReadingTransform, get_data_simultaneously, collate_time_adjust
from data_process.phoneme_encode import get_classes
from train_lipreading import make_model
from data_check import save_data_lipreading


# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def make_test_loader(cfg, data_root, mean_std_path):
    # classesを取得するために一旦学習用データを読み込む
    data_path = get_data_simultaneously(
        data_root=data_root,
        name=cfg.model.name,
    )
    classes = get_classes(data_path) 

    # テストデータを取得
    test_data_path = get_data_simultaneously(
        data_root=data_root,
        name=cfg.model.name,
    )

    # transform
    test_trans = LipReadingTransform(
        cfg=cfg,
        train_val_test="test",
    )

    # dataset
    test_dataset = LipReadingDataset(
        data_path=test_data_path,
        mean_std_path = mean_std_path,
        transform=test_trans,
        cfg=cfg,
        test=True,
        classes=classes,
    )

    # dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, test_dataset


def generate(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)
    classes_index = dataset.classes_index

    for batch in test_loader:
        model.eval()

        wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = batch
        lip, phoneme_index, data_len = lip.to(device), phoneme_index.to(device), data_len.to(device)

        phoneme_index_input = phoneme_index[:, :-1]

        phoneme_index_output = phoneme_index[:, 1:]
        
        with torch.no_grad():
            output = model(lip=lip, n_max_loop=int(phoneme_index_output.shape[-1] + 10))

        # ディレクトリ作成
        _save_path = save_path / label[0]
        os.makedirs(_save_path, exist_ok=True)

        # 保存
        save_data_lipreading(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            lip_mean=lip_mean,
            lip_std=lip_std,
            phoneme_index_output=phoneme_index_output,
            output=output,
            classes_index=classes_index,
        )
        

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # 口唇動画か顔かの選択
    lip_or_face = cfg.test.face_or_lip
    if lip_or_face == "face":
        data_root = cfg.test.face_pre_loaded_path
        mean_std_path = cfg.test.face_mean_std_path
    elif lip_or_face == "lip":
        data_root = cfg.test.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    elif lip_or_face == "lip_128128":
        data_root = cfg.test.lip_pre_loaded_path_128128
        mean_std_path = cfg.train.lip_mean_std_path_128128
    elif lip_or_face == "lip_9696":
        data_root = cfg.test.lip_pre_loaded_path_9696
        mean_std_path = cfg.train.lip_mean_std_path_9696
    elif lip_or_face == "lip_9696_time_only":
        data_root = cfg.test.lip_pre_loaded_path_9696_time_only
        mean_std_path = cfg.train.lip_mean_std_path_9696_time_only

    data_root = Path(data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

    print("--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")

    model = make_model(cfg, device)
    model_path = Path("~/lip2sp_pytorch/check_point/lipreading/lip_9696_time_only/2022:08:14_19-18-46/mspec80_230.ckpt").expanduser()
    model.load_state_dict(torch.load(str(model_path))['model'])

    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/result/default/train/lip_128128/2022:07:18_17-23-41/model_mspec80.pth")
    # model.load_state_dict(torch.load(str(model_path)))

    # 保存先
    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / lip_or_face / model_path.parents[0].name / model_path.stem
    os.makedirs(save_path, exist_ok=True)

    # Dataloader作成
    test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

    # generate
    generate(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        dataset=test_dataset,
        device=device,
        save_path=save_path,
    )
    

if __name__ == "__main__":
    main()