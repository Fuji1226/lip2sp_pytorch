import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/data_process").expanduser()))

import numpy as np
import hydra
from tqdm import tqdm
import csv
import pickle

from transform import load_data_for_npz_audio

debug = False
speaker = "F1" #これ挙動わからん
margin = 0
fps = 25
gray = True

csv_path = Path(f"/home/user/dataset/lip/data_split_csv/jvs.csv").expanduser()#!この辺バグる説ある
data_dir = Path(f"/home/user/dataset/jvs_ver1").expanduser()#!この辺バグる説有る
landmark_dir = Path(f"~/dataset/lip/landmark/{speaker}").expanduser()
dir_name = f"face_cropped_max_size_fps25_{margin}_{fps}"

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
        data_list = [[data_dir/"video/fps25/front/alldata"/f"{row[0]}.mp4", data_dir/"audio"/f"{row[0]}.wav", landmark_dir / f"{row[0]}.csv"] for row in reader]
    return data_list


def read_csv_gpt(csv_path, which_data):
    with open(str(csv_path), "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダー行をスキップ
        data_list = [
            [
                data_dir / "video/fps25/front/alldata" / f"{row[2]}.mp4",
                data_dir /f"{row[0]}"/ f"{row[1]}" /"wav24kHz16bit"/ f"{row[2]}.wav",
                landmark_dir / f"{row[2]}_front.csv"
            ]
            for row in reader if row[3] == which_data
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
            video_path, audio_path, landmark_path = data_list[i]
            #print(video_path)

            # 話者ラベル(F01_kablabとかです)
            audio_path.parents[0].name

            wav, feature = load_data_for_npz_audio(
                audio_path=audio_path,
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
                feature=feature,
            )

        except Exception as e: #例外処理-エラー出力
            print(f"error : {audio_path.stem}")
            print(e.__class__.__name__)
            print(e.args)
            print(e)
            print(f"{e.__class__.__name__}: {e}")

        if debug:
            break


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    """
    顔をやるか口唇切り取ったやつをやるかでpathを変更してください
    """

    cfg.model.gray = gray
    print(f"speaker = {speaker}, mode = {cfg.model.name}, gray = {cfg.model.gray}")

    train_data_list = read_csv_gpt(csv_path, "train")
    val_data_list = read_csv_gpt(csv_path, "val")
    test_data_list = read_csv_gpt(csv_path, "test")

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