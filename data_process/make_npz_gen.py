"""
face generationで合成したファイルのnpz
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import numpy as np
import torch
import hydra

try:
    from .transform_no_chainer import load_data_for_npz
except:
    from transform_no_chainer import load_data_for_npz


def get_dataset(data_root):
    items = []
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if Path(file).suffix == ".wav":
                audio_path = os.path.join(curdir, file)
                video_path = os.path.join(curdir, f"{Path(file).stem}_pred.mp4")

                if os.path.isfile(video_path) and os.path.isfile(audio_path):
                    items.append([video_path, audio_path])
    return items
            

def _save_data(items, len, cfg, data_save_path, mean_std_save_path, device, time_only):
    lip_mean = 0
    lip_std = 0
    feat_mean = 0
    feat_std = 0
    feat_add_mean = 0
    feat_add_std = 0

    for i in tqdm(range(len)):
        video_path, audio_path = items[i]
        video_path, audio_path = Path(video_path), Path(audio_path)
        
        speaker = video_path.parents[0].name

        wav, (lip, feature, feat_add, upsample), data_len = load_data_for_npz(
            video_path=video_path,
            audio_path=audio_path,
            cfg=cfg,
        )

        if cfg.model.name == "mspec80":
            assert feature.shape[-1] == 80
        elif cfg.model.name == "mspec40":
            assert feature.shape[-1] == 40
        elif cfg.model.name == "mspec60":
            assert feature.shape[-1] == 60
        elif cfg.model.name == "world":
            assert feature.shape[-1] == 29
        elif cfg.model.name == "world_melfb":
            assert feature.shape[-1] == 32
        
        assert feat_add.shape[-1] == 2
        
        # データの保存
        os.makedirs(os.path.join(data_save_path, speaker), exist_ok=True)
        np.savez(
            f"{data_save_path}/{speaker}/{audio_path.stem}_{cfg.model.name}",
            wav=wav,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            upsample=upsample,
            data_len=data_len,
        )


def save_data(train_data_root, train_data_save_path, train_mean_std_save_path, test_data_root, test_data_save_path, test_mean_std_save_path, cfg, device):
    train_items = get_dataset(train_data_root)
    n_data_train = len(train_items)
    _save_data(
        items=train_items,
        len=n_data_train,
        cfg=cfg,
        data_save_path=train_data_save_path,
        mean_std_save_path=train_mean_std_save_path,
        device=device,
        time_only=True,
    )

    test_items = get_dataset(test_data_root)
    n_data_test = len(test_items)
    _save_data(
        items=test_items,
        len=n_data_test,
        cfg=cfg,
        data_save_path=test_data_save_path,
        mean_std_save_path=test_mean_std_save_path,
        device=device,
        time_only=True
    )


@hydra.main(config_name="config", config_path=str(Path("~/lip2sp_pytorch/conf").expanduser()))
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    save_data(
        train_data_root=Path(cfg.train.gen_pre_loaded_path).expanduser(),
        trian_data_save_path=Path(cfg.train.gen_pre_loaded_path).expanduser(),
        train_mean_std_save_path=None,
        test_data_root=Path(cfg.test.gen_pre_loaded_path).expanduser(),
        test_data_save_path=Path(cfg.test.gen_pre_loaded_path).expanduser(),
        test_mean_std_save_path=None,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()