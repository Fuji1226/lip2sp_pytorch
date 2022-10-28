"""
train_val_test_split.pyでデータ分割を行った後に実行
動画や音響特徴量を事前に全て計算しておき,npz形式で保存しておくことでモデル学習時の計算時間を短縮します
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/data_process").expanduser()))

import numpy as np
import hydra
from tqdm import tqdm
import csv
import pickle

from transform_no_chainer import load_data_for_npz

debug = False
time_only = False
speaker = "F01_kablab"
margin = 0.3

if debug:
    if time_only:
        dirname = "lip_cropped_debug"
    else:
        dirname = "lip_cropped_st_debug"
else:
    if time_only:
        dirname = "lip_cropped"
    else:
        dirname = "lip_cropped_st"

csv_path = Path(f"~/dataset/lip/data_split_csv_{margin}").expanduser()
lip_train_data_path = Path(f"~/dataset/lip/np_files/{dirname}_{margin}/train").expanduser()
lip_stat_path = Path(f"~/dataset/lip/np_files/{dirname}_{margin}/stat").expanduser()
lip_val_data_path = Path(f"~/dataset/lip/np_files/{dirname}_{margin}/val").expanduser()
lip_test_data_path = Path(f"~/dataset/lip/np_files/{dirname}_{margin}/test").expanduser()


def read_csv(csv_path, which_data):
    with open(str(csv_path / which_data / f"{speaker}.csv"), "r") as f:
        reader = csv.reader(f)
        data_list = [row for row in reader]
    return data_list
    

def save_data(data_list, len, cfg, data_save_path, which_data, stat_path=None):
    """
    データ，平均，標準偏差の保存
    話者ごとに行うことを想定してます
    """
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []
    feat_add_mean_list = []
    feat_add_var_list = []
    feat_add_len_list = []

    print(f"save {which_data}")
    for i in tqdm(range(len)):
        try:
            video_path, audio_path = data_list[i]
            video_path, audio_path = Path(video_path), Path(audio_path)

            # 話者ラベル(F01_kablabとかです)
            speaker = audio_path.parents[1].name

            wav, (lip, feature, feat_add, upsample), data_len = load_data_for_npz(
                video_path=video_path,
                audio_path=audio_path,
                cfg=cfg,
            )

            if cfg.model.name == "mspec80":
                assert feature.shape[1] == 80
            elif cfg.model.name == "world_melfb":
                assert feature.shape[1] == 32
            
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

            if which_data == "train":
                # 時間方向のみ
                if time_only:
                    lip_mean_list.append(np.mean(lip, axis=-1))
                    lip_var_list.append(np.var(lip, axis=-1))
                    lip_len_list.append(lip.shape[-1])
                # 時空間両方
                else:
                    lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
                    lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
                    lip_len_list.append(lip.shape[-1])

                feat_mean_list.append(np.mean(feature, axis=0))
                feat_var_list.append(np.var(feature, axis=0))
                feat_len_list.append(feature.shape[0])
                feat_add_mean_list.append(np.mean(feat_add, axis=0))
                feat_add_var_list.append(np.var(feat_add, axis=0))
                feat_add_len_list.append(feat_add.shape[0])

        except:
            print(f"error : {audio_path.stem}")
        
        if debug:
            break

    if which_data == "train":
        os.makedirs(os.path.join(stat_path, speaker), exist_ok=True)
        filename = [
            "lip_mean_list", "lip_var_list", "lip_len_list", 
            "feat_mean_list", "feat_var_list", "feat_len_list", 
            "feat_add_mean_list", "feat_add_var_list", "feat_add_len_list"
        ]
        data_list = [
            lip_mean_list, lip_var_list, lip_len_list,
            feat_mean_list, feat_var_list, feat_len_list,
            feat_add_mean_list, feat_add_var_list, feat_add_len_list,
        ]
        for name, data in zip(filename, data_list):
            with open(f"{stat_path}/{speaker}/{name}_{cfg.model.name}.bin", "wb") as p:
                pickle.dump(data, p)

        print("save stat")


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    """
    顔をやるか口唇切り取ったやつをやるかでpathを変更してください
    """
    print(f"speaker = {speaker}, mode = {cfg.model.name}, time_only = {time_only}")

    train_data_list = read_csv(csv_path, "train")
    val_data_list = read_csv(csv_path, "val")
    test_data_list = read_csv(csv_path, "test")
    
    print(f"\nall data ratio")
    print(f"train_data : {len(train_data_list)}, val_data : {len(val_data_list)}, test_data : {len(test_data_list)}")

    save_data(
        data_list=train_data_list,
        len=len(train_data_list),
        cfg=cfg,
        data_save_path=str(lip_train_data_path),
        which_data="train",
        stat_path=str(lip_stat_path),
    )

    save_data(
        data_list=val_data_list,
        len=len(val_data_list),
        cfg=cfg,
        data_save_path=str(lip_val_data_path),
        which_data="val",
    )

    save_data(
        data_list=test_data_list,
        len=len(test_data_list),
        cfg=cfg,
        data_save_path=str(lip_test_data_path),
        which_data="test",
    )


if __name__ == "__main__":
    main()