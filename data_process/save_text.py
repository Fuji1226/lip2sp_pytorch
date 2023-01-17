from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pyopenjtalk
import csv
import shutil


def save_utt(df_data, df_atr, df_basic, df_balanced, save_dir):
    for i in tqdm(range(len(df_data))):
        data_name = df_data.iloc[i].values[0]
        utt = None

        if Path(str(save_dir / f"{data_name}.csv")).exists():
            continue

        if "ATR" in data_name:
            for j in range(len(df_atr)):        
                if df_atr.iloc[j].utt_num in data_name:
                    utt = df_atr.iloc[j]
        elif "BASIC5000" in data_name:
            for j in range(len(df_basic)):        
                if df_basic.iloc[j].utt_num in data_name:
                    utt = df_basic.iloc[j]
        elif "balanced" in data_name:
            for j in range(len(df_balanced)):        
                if df_balanced.iloc[j].utt_num in data_name:
                    utt = df_balanced.iloc[j]
        
        header = list(utt.index.values)
        value = list(utt.values)
    
        if utt is not None:
            with open(str(save_dir / f"{data_name}.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(value)
        else:
            continue


def main():
    csv_dir = Path("~/lip2sp_pytorch/csv").expanduser()
    atr = csv_dir / "ATR503.csv"
    basic = csv_dir / "BASIC5000.csv"
    balanced = csv_dir / "balanced.csv"
    df_atr = pd.read_csv(str(atr))
    df_basic = pd.read_csv(str(basic))
    df_balanced = pd.read_csv(str(balanced))
    
    speaker = "F02_kablab"
    print(f"speaker = {speaker}")
    data_split_csv_dir = Path("~/dataset/lip/data_split_csv").expanduser()
    data_split_csv_dir = data_split_csv_dir / speaker
    train = data_split_csv_dir / "train.csv"
    val = data_split_csv_dir / "val.csv"
    test = data_split_csv_dir / "test.csv"
    df_train = pd.read_csv(str(train), header=None)
    df_val = pd.read_csv(str(val), header=None)
    df_test = pd.read_csv(str(test), header=None)

    utt_save_dir = Path("~/dataset/lip/utt").expanduser()
    utt_save_dir_train = utt_save_dir / "train" / speaker
    utt_save_dir_val = utt_save_dir / "val" / speaker
    utt_save_dir_test = utt_save_dir / "test" / speaker
    utt_save_dir_train.mkdir(parents=True, exist_ok=True)
    utt_save_dir_val.mkdir(parents=True, exist_ok=True)
    utt_save_dir_test.mkdir(parents=True, exist_ok=True)

    print("train")
    save_utt(df_train, df_atr, df_basic, df_balanced, utt_save_dir_train)

    print("val")
    save_utt(df_val, df_atr, df_basic, df_balanced, utt_save_dir_val)

    print("test")
    save_utt(df_test, df_atr, df_basic, df_balanced, utt_save_dir_test)


def load_test():
    speaker = "F01_kablab"
    utt_save_dir = Path("~/dataset/lip/utt").expanduser()
    utt_save_dir_train = utt_save_dir / "train" / speaker
    utt_save_dir_val = utt_save_dir / "val" / speaker
    utt_save_dir_test = utt_save_dir / "test" / speaker

    data = sorted(list(utt_save_dir_train.glob("*.csv")))
    for path in data:
        df = pd.read_csv(str(path))
        text = df.pronounce.values[0]
        phoneme = pyopenjtalk.g2p(text)
        print(text)
        print(phoneme)
        print("")


def change_npz_name():
    speaker_list = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab"]
    dirname = "lip_cropped_0.8_50_gray"
    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        data_dir = Path(f"~/dataset/lip/np_files/{dirname}").expanduser()
        save_dir = Path(f"~/dataset/lip/np_files/{dirname}/mspec80").expanduser()

        data_seg_list = ["train", "val", "test"]
        for data_seg in data_seg_list:
            print(f"{data_seg}")
            data_dir_seg = data_dir / data_seg / speaker
            data_path_seg = sorted(list(data_dir_seg.glob("*.npz")))
            
            for path in data_path_seg:
                filename = path.stem
                filename = "_".join(filename.split("_")[:-1])

                save_dir_seg = save_dir / data_seg / speaker
                save_dir_seg.mkdir(parents=True, exist_ok=True)
                new_path = save_dir_seg / f"{filename}.npz"
                path.rename(str(new_path))


def change_npz_dir():
    speaker_list = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab"]
    dirname = "face_aligned_0_50"
    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        data_dir = Path(f"~/dataset/lip/np_files/{dirname}/mspec80").expanduser()
        save_dir = Path(f"~/dataset/lip/np_files/{dirname}").expanduser()

        data_seg_list = ["train", "val", "test"]
        for data_seg in data_seg_list:
            print(data_seg)
            data_dir_seg = data_dir / data_seg / speaker
            data_path_seg = sorted(list(data_dir_seg.glob("*.npz")))

            for path in tqdm(data_path_seg):
                save_dir_seg = save_dir / data_seg / speaker / "mspec80"
                save_dir_seg.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(save_dir_seg / f"{path.stem}.npz"))


if __name__ == "__main__":
    main()
    # load_test()
    # change_npz_name()
    # change_npz_dir()