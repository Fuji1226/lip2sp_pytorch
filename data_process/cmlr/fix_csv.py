from pathlib import Path
import os

import csv
import pandas as pd
import random

is_fixed = True


def replace_underbar_slash(df):
    fixed_path = []
    for i in range(len(df)):
        path = df.iloc[i].path.replace("_", "/", 1)
        fixed_path.append([path])
    return fixed_path


def main():
    data_dir = Path("~/cmlr").expanduser()

    # 余計なアンダーバーをスラッシュに変換
    if is_fixed == False:
        train_df = pd.read_csv(str(data_dir / "train.csv"), header=None, names=["path"])
        test_df = pd.read_csv(str(data_dir / "test.csv"), header=None, names=["path"])
        fixed_path_list = [replace_underbar_slash(train_df), replace_underbar_slash(test_df)]
        save_csv_file_name = ["train_fixed.csv", "test_fixed.csv"]
        for fixed_path, file_name in zip(fixed_path_list, save_csv_file_name):
            with open(str(data_dir / file_name), "w") as f:
                writer = csv.writer(f)
                for path in fixed_path:
                    writer.writerow(path)

    train_df = pd.read_csv(str(data_dir / "train_fixed.csv"), header=None, names=["path"])
    test_df = pd.read_csv(str(data_dir / "test_fixed.csv"), header=None, names=["path"])

    data_path = []
    speaker_list = [f"s{i + 1}" for i in range(11)]
    for speaker in speaker_list:
        spk_data_dir = data_dir / speaker

        for curdir, dirs, files in os.walk(spk_data_dir):
            for file in files:
                file = Path(curdir, file)
                if file.suffix == ".wav":
                    file = f"{file.parents[1].name}/{file.parents[0].name}/{file.stem}"
                    data_path.append(file)

    # val.csvがバグっているのでtrain.csvとtest.csvに含まれないデータを探す
    df_list = [train_df, test_df]
    for df in df_list:
        for path in df.path:
            data_path.remove(path)

    data_path = random.sample(data_path, len(data_path))

    with open(str(data_dir / "val_fixed.csv"), "w") as f:
        writer = csv.writer(f)
        for path in data_path:
            writer.writerow([path])


if __name__ == "__main__":
    main()