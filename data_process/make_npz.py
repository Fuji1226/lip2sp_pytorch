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

from transform import load_data_for_npz

debug = False
speaker = "M01_kablab"
margin = 0
fps = 50
gray = True

csv_path = Path(f"~/dataset/lip/data_split_csv").expanduser()
data_dir = Path(f"~/dataset/lip/cropped_max_size/{speaker}").expanduser()
landmark_dir = Path(f"~/dataset/lip/landmark_cropped_max_size/{speaker}").expanduser()
dir_name = f"face_cropped_max_size_{margin}_{fps}"

if gray:
    dir_name = f"{dir_name}_gray"

if debug:
    dir_name = f"{dir_name}_debug"

lip_train_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/train").expanduser()
lip_val_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/val").expanduser()
lip_test_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/test").expanduser()


def read_csv(csv_path, which_data):
    with open(str(csv_path / f"{which_data}.csv"), "r") as f:
        reader = csv.reader(f)
        data_list = [[data_dir / f"{row[0]}.mp4", data_dir / f"{row[0]}.wav", landmark_dir / f"{row[0]}.csv"] for row in reader]
    return data_list
    

def save_data(data_list, len, cfg, data_save_path, which_data):
    """
    データ，平均，標準偏差の保存
    話者ごとに行うことを想定してます
    """
    print(f"save {which_data}")
    for i in tqdm(range(len)):
        try:
            video_path, audio_path, landmark_path = data_list[i]

            # 話者ラベル(F01_kablabとかです)
            speaker = audio_path.parents[0].name

            wav, lip, feature, feat_add, upsample, data_len, landmark = load_data_for_npz(
                video_path=video_path,
                audio_path=audio_path,
                landmark_path=landmark_path,
                cfg=cfg,
            )

            if cfg.model.name == "mspec80":
                assert feature.shape[1] == 80
            elif cfg.model.name == "world_melfb":
                assert feature.shape[1] == 32
            
            # データの保存
            _data_save_path = data_save_path / speaker / cfg.model.name
            _data_save_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(_data_save_path / audio_path.stem),
                wav=wav,
                lip=lip,
                feature=feature,
                feat_add=feat_add,
                landmark=landmark,
                upsample=upsample,
                data_len=data_len,
            )

        except:
            print(f"error : {audio_path.stem}")
        
        if debug:
            break


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    """
    顔をやるか口唇切り取ったやつをやるかでpathを変更してください
    """
    cfg.model.gray = gray
    print(f"speaker = {speaker}, mode = {cfg.model.name}, gray = {cfg.model.gray}")

    train_data_list = read_csv(csv_path, "train_all")
    val_data_list = read_csv(csv_path, "val")
    test_data_list = read_csv(csv_path, "test")
    
    print(f"\nall data ratio")
    print(f"train_data : {len(train_data_list)}, val_data : {len(val_data_list)}, test_data : {len(test_data_list)}")

    save_data(
        data_list=train_data_list,
        len=len(train_data_list),
        cfg=cfg,
        data_save_path=lip_train_data_path,
        which_data="train",
    )

    save_data(
        data_list=val_data_list,
        len=len(val_data_list),
        cfg=cfg,
        data_save_path=lip_val_data_path,
        which_data="val",
    )

    save_data(
        data_list=test_data_list,
        len=len(test_data_list),
        cfg=cfg,
        data_save_path=lip_test_data_path,
        which_data="test",
    )


if __name__ == "__main__":
    main()