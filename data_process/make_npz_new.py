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
import pandas as pd
import argparse

from transform import load_data_for_npz_new

debug = False


def read_csv(csv_path, which_data, video_dir, audio_dir):
    df = pd.read_csv(str(csv_path / f'{which_data}.csv'))
    filename_list = df['filename'].to_list()
    data_list = [
        [video_dir / f'{filename}.mp4', audio_dir / f'{filename}.wav']
        for filename in filename_list
    ]
    return data_list
    

def save_data(data_list, len, cfg, data_save_path, which_data):
    """
    データ，平均，標準偏差の保存
    話者ごとに行うことを想定してます
    """
    print(f"save {which_data}")
    for i in tqdm(range(len)):
        try:
            video_path, audio_path = data_list[i]

            # 話者ラベル(F01_kablabとかです)
            speaker = audio_path.parents[0].name

            wav, lip, feature, feat_add, upsample, data_len = load_data_for_npz_new(
                video_path=video_path,
                audio_path=audio_path,
                cfg=cfg,
            )
            _data_save_path = data_save_path / speaker / cfg.model.name
            _data_save_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(_data_save_path / audio_path.stem),
                wav=wav,
                lip=lip,
                feature=feature,
            )

        except:
            print(f"error : {audio_path.stem}")
        
        if debug:
            break


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    for speaker in cfg.train.npz_process_speaker_list:
        margin = 0
        fps = 25
        gray = True

        csv_path = Path(f"~/dataset/lip/data_split_csv_fix").expanduser()
        video_dir = Path(f"~/dataset/lip/avhubert_preprocess_fps25/{speaker}").expanduser()
        audio_dir = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
        dir_name = f"avhubert_preprocess_fps25"

        if gray:
            dir_name = f"{dir_name}_gray"
        if debug:
            dir_name = f"{dir_name}_debug"

        lip_train_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/train").expanduser()
        lip_val_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/val").expanduser()
        lip_test_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/test").expanduser()
            
        cfg.model.gray = gray
        print(f"speaker = {speaker}, mode = {cfg.model.name}, gray = {cfg.model.gray}")

        train_data_list = read_csv(csv_path, "train", video_dir, audio_dir)
        val_data_list = read_csv(csv_path, "val", video_dir, audio_dir)
        test_data_list = read_csv(csv_path, "test", video_dir, audio_dir)
        
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