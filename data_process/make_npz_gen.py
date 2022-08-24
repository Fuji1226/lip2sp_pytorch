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
    test_items = []
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if Path(file).suffix == ".wav":
                audio_path = os.path.join(curdir, file)
                video_path = os.path.join(curdir, f"{Path(file).stem}_pred.mp4")

                if os.path.isfile(video_path) and os.path.isfile(audio_path):
                    test_items.append([video_path, audio_path])
    return test_items
            

def save_data_test(items, len, cfg, data_save_path, mean_std_save_path, device, time_only):
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

        # lip = torch.from_numpy(lip).to(device)
        # feature = torch.from_numpy(feature).to(device)
        # feat_add = torch.from_numpy(feat_add).to(device)

        # if time_only:
        #     # 時間方向のみの平均、標準偏差を計算
        #     print("time only")
        #     lip_mean += torch.mean(lip.float(), dim=3)
        #     lip_std += torch.std(lip.float(), dim=3)
        # else:
        #     # 時間、空間方向両方の平均、標準偏差を計算
        #     lip_mean += torch.mean(lip.float(), dim=(1, 2, 3))
        #     lip_std += torch.std(lip.float(), dim=(1, 2, 3))
        
    #     feat_mean += torch.mean(feature, dim=0)
    #     feat_std += torch.std(feature, dim=0)
    #     feat_add_mean += torch.mean(feat_add, dim=0)
    #     feat_add_std += torch.std(feat_add, dim=0)

    # # データ全体の平均、分散を計算 (C,) チャンネルごと
    # lip_mean /= len     
    # lip_std /= len      
    # feat_mean /= len    
    # feat_std /= len     
    # feat_add_mean /= len
    # feat_add_std /= len

    # lip_mean = lip_mean.to('cpu').detach().numpy().copy()
    # lip_std = lip_std.to('cpu').detach().numpy().copy()
    # feat_mean = feat_mean.to('cpu').detach().numpy().copy()
    # feat_std = feat_std.to('cpu').detach().numpy().copy()
    # feat_add_mean = feat_add_mean.to('cpu').detach().numpy().copy()
    # feat_add_std = feat_add_std.to('cpu').detach().numpy().copy()
    
    # os.makedirs(os.path.join(mean_std_save_path, speaker), exist_ok=True)
    # np.savez(
    #     f"{mean_std_save_path}/{speaker}/test_{cfg.model.name}",
    #     lip_mean=lip_mean, 
    #     lip_std=lip_std, 
    #     feat_mean=feat_mean, 
    #     feat_std=feat_std, 
    #     feat_add_mean=feat_add_mean, 
    #     feat_add_std=feat_add_std,
    # )


def save_data(data_root, test_data_save_path, test_mean_std_save_path, cfg, device):
    test_items = get_dataset(data_root)
    n_data_test = len(test_items)

    save_data_test(
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
        data_root=Path(cfg.test.gen_pre_loaded_path).expanduser(),
        test_data_save_path=Path(cfg.test.gen_pre_loaded_path).expanduser(),
        test_mean_std_save_path=None,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()