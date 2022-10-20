import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.wavenet import WaveNet
from dataset_wavenet import WaveNetDataset, WaveNetTransform
from dataset.dataset_npz import get_datasets
from data_process.mulaw import inv_mulaw_quantize
from wavenet.train import make_model
from wavenet.data_check import save_data

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def make_test_loader(cfg, data_root, mean_std_path):
    # パス取得
    data_path = get_datasets(
        data_root=data_root,
        name=cfg.model.name,
    )

    # transform
    test_trans = WaveNetTransform(cfg, "test")

    # dataset
    test_dataset = WaveNetDataset(
        data_path=data_path,
        mean_std_path=mean_std_path,
        transform=test_trans,
        cfg=cfg,
        test=True,
    )

    # dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    return test_loader


def generate(model, test_loader, device, save_path, cfg):
    model.eval()

    for batch in test_loader:
        wav, feature, label = batch
        wav, feature = wav.to(device), feature.to(device)
        print(f"wav = {wav.shape}")
        print(f"feature = {feature.shape}")

        with torch.no_grad():
            wav_gen = model.inference(feature)

        # wav_genの最大値のインデックスを取得
        wav_gen = wav_gen.max(dim=1)[1].float().cpu().numpy().reshape(-1)

        # mulawアルゴリズムを適用しているので，逆変換
        wav = wav.cpu().detach().numpy()
        wav = inv_mulaw_quantize(wav, model.out_channels - 1)
        wav_gen = inv_mulaw_quantize(wav_gen, model.out_channels - 1)
        print(f"wav = {wav.shape}")
        print(wav)
        print(f"wav_gen = {wav_gen.shape}")
        print(wav_gen)
        wav = wav.squeeze(0)
        # wav_gen = wav_gen.squeeze(0)

        _save_path = save_path / label[0]
        os.makedirs(_save_path, exist_ok=True)
        save_data(cfg, wav, wav_gen, _save_path)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # path
    if cfg.test.check_1to5:
        data_root = cfg.test.data_root_1to5
    else:
        data_root = cfg.test.data_root
    mean_std_path = cfg.test.mean_std_path

    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")

    # model
    model = make_model(cfg, device)

    # load parameter
    model_path = Path("~/lip2sp_pytorch/wavenet/check_point/2022:07:22_18-02-48/mspec80_190.ckpt").expanduser()
    model.load_state_dict(torch.load(model_path)["model"])

    # 保存先
    save_path = Path(cfg.test.generate_save_path)
    save_path = save_path / model_path.parents[0].name / model_path.stem
    os.makedirs(save_path, exist_ok=True)

    # dataloader
    test_loader = make_test_loader(cfg, data_root, mean_std_path)

    # generate
    generate(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path=save_path,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()