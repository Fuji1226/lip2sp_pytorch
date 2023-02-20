"""
f01_kablabの追加データを統合
データ量増強の効果を見たいので,一旦全て学習データとして使用
"""
from pathlib import Path
import os
import random
import csv
from tqdm import tqdm
import pandas as pd

random.seed(777)

# speakerのみ変更してください
speaker = "F01_kablab_20220930"
data_dir = Path(f"~/dataset/lip/cropped_fps25/{speaker}").expanduser()
save_dir = Path(f"~/dataset/lip/data_split_csv").expanduser()
corpus = ["ATR", "balanced", "BASIC5000"]
train_ratio = 0.95


def write_csv(data_list, which_data):
    csv_save_path = save_dir / speaker
    os.makedirs(csv_save_path, exist_ok=True)

    with open(str(csv_save_path / f"{which_data}.csv"), "w") as f:
        writer = csv.writer(f)
        for data in tqdm(data_list):
            writer.writerow(data)


def main():
    data_path = sorted(list(data_dir.glob("*.wav")))
    data_path = [data_path[i].stem for i in range(len(data_path))]
    df_data_path = pd.DataFrame(data_path)
    data_split_csv = pd.read_csv(str(save_dir / "train.csv"), header=None)
    df_merge = pd.concat([data_split_csv, df_data_path])
    df_merge.to_csv(str(save_dir / "train_all.csv"), index=False, header=None)
    df_merge = pd.read_csv(str(save_dir / "train_all.csv"), header=None)
    breakpoint()


if __name__ == "__main__":
    main()