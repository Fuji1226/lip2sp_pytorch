"""
juliusを用いて音素アラインメントを行った時にミスしているデータがたまにあるので,それを取り除いて使えるデータをコピーする
"""

import os
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv


def read_csv(csv_path, speaker, which_data, lab_dir):
    with open(str(csv_path / speaker / f"{which_data}.csv"), "r") as f:
        reader = csv.reader(f)
        data_list = [lab_dir / f"{row[0]}.lab" for row in reader]
    return data_list


def main():
    speaker = "F01_kablab"
    lab_dir = Path("~/dataset/segmentation-kit/wav").expanduser()

    csv_path = Path(f"~/dataset/lip/data_split_csv").expanduser()
    save_dir = Path("~/dataset/lip/np_files/face_aligned_0_50_gray").expanduser()
    data_seg_list = ["train", "val", "test"]

    for data_seg in data_seg_list:
        save_dir_seg = save_dir / data_seg / speaker
        save_dir_seg.mkdir(parents=True, exist_ok=True)
        data_list = read_csv(csv_path, speaker, data_seg, lab_dir)
        for data in data_list:
            with open(str(data), "r") as f:
                da = f.read()

            da = da.replace("\n", " ").split(" ")
            da = da[2::3]
            if da != []:
                shutil.copy(str(data), save_dir_seg)

    
if __name__ == "__main__":
    main()