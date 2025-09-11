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

#!データパスの指定！

debug = False
speaker = "M02_kablab"
margin = 0
fps = 25
gray = True

csv_path = Path(f"/home/user/dataset/lip/data_split_csv/kab2022.csv").expanduser()
a_data_dir = Path(f"/home/user/dataset/{speaker}").expanduser()
v_data_dir = Path(f"/home/user/dataset/lip/avhubert_preprocess_fps25/{speaker}").expanduser()
landmark_dir = Path(f"~/dataset/lip/landmark/{speaker}").expanduser()
dir_name = f"face_cropped_max_size_fps25_{margin}_{fps}"

if speaker == "F1":
    csv_path = Path(f"/home/user/2HEAVD/ITA_text/{speaker}").expanduser()
    a_data_dir = Path(f"/home/user/2HEAVD/ITA_text/F1/audio/alldata").expanduser()
#!データパスの指定！


if gray:
    dir_name = f"{dir_name}_gray"

if debug:
    dir_name = f"{dir_name}_debug"

lip_train_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/train").expanduser()
lip_val_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/val").expanduser()
lip_test_data_path = Path(f"~/dataset/lip/np_files/{dir_name}/test").expanduser()


def read_csv_katsu(csv_path, which_data):
    with open(str(csv_path / f"{which_data}.csv"), "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダー行をスキップ
        data_list = [[v_data_dir/f"{row[0]}_front.mp4", a_data_dir/f"{row[0]}.wav", landmark_dir / f"{row[0]}_front.csv"] for row in reader]
    return data_list


def read_csv_gpt(csv_path, which_data):
    with open(str(csv_path), "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダー行をスキップ
        data_list = [
            [
                a_data_dir / "video/fps25/front/alldata" / f"{row[2]}.mp4",
                v_data_dir /f"{row[0]}"/ f"{row[1]}" /"wav24kHz16bit"/ f"{row[2]}.wav",
                landmark_dir / f"{row[2]}.csv"
            ]
            for row in reader if row[3] == which_data
        ]
    return data_list

def read_csv_kab2022(csv_path, which_data, which_speaker):
    """
    [speaker, data, filename, data_split, label] 形式のCSVから
    動画・音声・ランドマークのパスリストを返す
    """
    data_list = []
    with open(str(csv_path), "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダー行をスキップ
        for row in reader:
            speaker, data, filename, data_split, label = row
            if data_split != which_data:
                continue
            if speaker != which_speaker:
                continue
            video_path = v_data_dir / f"{filename}.mp4"
            audio_path = a_data_dir / "wav" / f"{filename}.wav"
            landmark_path = landmark_dir / f"{filename}.csv"
            data_list.append([video_path, audio_path, landmark_path])
    return data_list
#data_dir = Path(f"/home/user/dataset/kab2022").expanduser()


def save_data(data_list, len, cfg, data_save_path, which_data):
    """
    データ，平均，標準偏差の保存
    話者ごとに行うことを想定してます
    """
    print(f"save {which_data}")
    for i in tqdm(range(len)):
        try:
            video_path, audio_path, landmark_path = data_list[i]
            if debug:
                print(video_path)
                print(audio_path)
                print(landmark_path)

            # 話者ラベル(F01_kablabとかです)
            audio_path.parents[0].name
            _data_save_path = data_save_path / speaker / cfg.model.name

            #保存先にすでに[_data_save_path/audio_path.stem.npz]ファイルがある場合はスキップ
            if (_data_save_path / audio_path.stem).with_suffix('.npz').exists():
                if debug:
                    print(f"skip: {audio_path.stem}")
                continue  # 何もせず次へ

            #ファイルが無い場合のみデータ生成・保存
            wav, lip, feature, feat_add, upsample, data_len, landmark = load_data_for_npz(
                video_path=video_path,
                audio_path=audio_path,
                landmark_path=landmark_path,
                cfg=cfg,
            )
            if debug:
                print(wav.shape)
                print(lip.shape)
                print(feature.shape)
                print(feat_add.shape)
                print(upsample)
                print(data_len)
                print(landmark.shape)
                breakpoint()

            #if cfg.model.name == "mspec80":
            assert feature.shape[1] == 80
            #elif cfg.model.name == "world_melfb":
                #assert feature.shape[1] == 32

            # データの保存
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

    if speaker == "F1":
        train_data_list = read_csv_katsu(csv_path, "train")
        val_data_list = read_csv_katsu(csv_path, "val")
        test_data_list = read_csv_katsu(csv_path, "test")

    else:
        train_data_list = read_csv_kab2022(csv_path, "train", speaker)
        val_data_list = read_csv_kab2022(csv_path, "val", speaker)
        test_data_list = read_csv_kab2022(csv_path, "test", speaker)

    print(f"\nall data ratio")
    print(f"train_data : {len(train_data_list)}, val_data : {len(val_data_list)}, test_data : {len(test_data_list)}")


    save_data(
        data_list=train_data_list,
        len=len(train_data_list),
        cfg=cfg,
        data_save_path=lip_train_data_path,
        which_data="train",
    )
    if debug:
        breakpoint()

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