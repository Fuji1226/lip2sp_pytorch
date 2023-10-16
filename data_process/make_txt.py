"""
juliusを用いた音素アラインメントを行うための前処理
csvファイルからテキストを取得し,一つずつtxtファイルに変換
発話内容に対応したwavファイルも取得し,同じディレクトリに保存する
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil


save_path = Path("~/dataset/segmentation-kit/wav").expanduser()   # 必ずここで
error_path = Path("~/dataset/segmentation-kit/failed_data.txt").expanduser()


def get_wav_path(data_root):
    paths = []
    for curdir, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".wav"):
                if "_norm" in Path(file).stem:
                    continue
                else:
                    path = os.path.join(curdir, file)
                    if os.path.isfile(path):
                        paths.append(path)
    return paths


def main():
    os.makedirs(save_path, exist_ok=True)

    # csvファイルまでのパスを取得
    data_root_csv = Path("~/lip2sp_pytorch/csv").expanduser()
    data_path_csv = list(data_root_csv.iterdir())

    # wavファイルまでのパスを取得
    data_root_wav = Path("~/dataset/lip/cropped/F01_kablab").expanduser()
    data_path_wav = get_wav_path(data_root_wav)

    # csvファイルの読み込み
    data = []
    for path in data_path_csv:
        data.append(pd.read_csv(path))
    
    # utt_numを取得
    utt_num = []
    for d in data:
        utt_num.append(d["utt_num"])

    # pronounceを取得
    pronounce = []
    for d in data:
        pronounce.append(d["pronounce"])
    
    for each_u, each_p in zip(utt_num, pronounce):
        for u, p in zip(each_u, each_p):
            # 句読点を置換
            p = p.replace("。", "")
            p = p.replace("、", " sp ")

            # 発話内容に対応したwavファイルを取得
            correspond_wav = []
            for wav in data_path_wav:
                if f"{u}_" in Path(wav).stem:
                    correspond_wav.append(wav)
                    label = Path(wav).stem

            if len(correspond_wav) != 1:
                with open(str(error_path), "a") as f:
                    f.write(f"{u}\n")
                continue

            sp = save_path / f"{label}.txt"
            with open(str(sp), "w") as f:
                f.write(str(p))

            shutil.copy(correspond_wav[-1], save_path)


if __name__ == "__main__":
    main()