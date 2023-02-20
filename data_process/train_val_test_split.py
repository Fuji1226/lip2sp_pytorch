"""
テスト用 : ATR503のJセット(53文)
学習用 : 残り全部の95%
検証用 : 残り全部の5%
"""

from pathlib import Path
import os
import random
import hydra
import csv
from tqdm import tqdm

random.seed(777)

# speakerのみ変更してください
speaker = "F01_kablab"
data_dir = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
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


def get_dataset():
    atr_train_val_data_list = []
    atr_test_data_list = []
    basic5000_train_val_data_list = []
    basic5000_test_data_list = []
    balanced_train_val_data_list = []
    balanced_test_data_list = []

    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            file = Path(curdir, file)
            if file.suffix == ".wav":
                if "ATR" in file.stem:
                    if "_j" in file.stem:
                        atr_test_data_list.append([file.stem])
                    else:
                        atr_train_val_data_list.append([file.stem])

                elif "BASIC5000" in file.stem:
                    basic5000_train_val_data_list.append([file.stem])

                elif "balanced" in file.stem:
                    balanced_train_val_data_list.append([file.stem])

        train_val_data_list = atr_train_val_data_list + basic5000_train_val_data_list + balanced_train_val_data_list
        train_val_data_list = random.sample(train_val_data_list, len(train_val_data_list))
        num_train_data = int(len(train_val_data_list) * train_ratio)

        train_data_list = train_val_data_list[:num_train_data]
        val_data_list = train_val_data_list[num_train_data:]
        test_data_list = atr_test_data_list + basic5000_test_data_list + balanced_test_data_list

        print(f"train : {len(train_data_list)}, val : {len(val_data_list)}, test : {len(test_data_list)}")

    return train_data_list, val_data_list, test_data_list


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    print(f"speaker = {speaker}")
    print("\nget dataset")
    train_data_list, val_data_list, test_data_list = get_dataset()

    print("\nwrite csv")
    write_csv(train_data_list, "train")
    write_csv(val_data_list, "val")
    write_csv(test_data_list, "test")
    

if __name__ == "__main__":
    main()