"""
lip_croppedを作り,make_corpus_dir.pyでコーパスごとのディレクトリに分けた後で使用
データ全体を学習用,検証用,テスト用データに分割し,それらのパスをcsv形式で保存しておく

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


# speakerのみ変更してください
speaker = "M01_kablab"
margin = 0.3
lip_path = Path(f"~/dataset/lip/lip_cropped_{margin}/{speaker}").expanduser()
save_path = Path(f"~/dataset/lip/data_split_csv_{margin}").expanduser()
corpus = ["ATR", "balanced", "BASIC5000"]
train_size = 0.95


def get_dataset(data_root, cfg):
    train_data_list = []
    val_data_list = []
    test_data_list = []

    for co in corpus:
        data_list_co = []
        test_data_list_co = []
        corpus_dir = data_root / co

        for curdir, dir, files in os.walk(corpus_dir):
            for file in files:
                if file.endswith(".wav"):
                    # ATRのjセットをテストデータにするので，ここで分けます
                    if '_j' in Path(file).stem:
                        audio_path = os.path.join(curdir, file)
                        video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
                        if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            test_data_list_co.append([video_path, audio_path])
                    else:
                        audio_path = os.path.join(curdir, file)
                        video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
                        if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            data_list_co.append([video_path, audio_path])

            # 一度ランダムに並び替えることでランダムサンプリングを行う
            data_list_co = random.sample(data_list_co, len(data_list_co))
            train_data_size = int(len(data_list_co) * train_size)

            train_data = data_list_co[:train_data_size]
            val_data = data_list_co[train_data_size:]
            test_data = test_data_list_co

            print(f"corpus = {co}")
            print(f"train : {len(train_data)}, val : {len(val_data)}, test : {len(test_data)}")

            train_data_list += train_data
            val_data_list += val_data
            test_data_list += test_data

    return train_data_list, val_data_list, test_data_list


def write_csv(data_list, which_data):
    csv_save_path = save_path / f"{which_data}"
    os.makedirs(csv_save_path, exist_ok=True)

    with open(str(csv_save_path / f"{speaker}.csv"), "w") as f:
        writer = csv.writer(f)
        for data in tqdm(data_list):
            writer.writerow(data)


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    print(f"speaker = {speaker}")
    print("\nget dataset")
    train_data_list, val_data_list, test_data_list = get_dataset(lip_path, cfg)

    print("\nwrite csv")
    write_csv(train_data_list, "train")
    write_csv(val_data_list, "val")
    write_csv(test_data_list, "test")
    

if __name__ == "__main__":
    main()