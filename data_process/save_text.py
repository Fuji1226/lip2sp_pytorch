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


def save_utt_raw(df, which_corpus, save_dir):
    for i in tqdm(range(len(df))):
        utt = df.iloc[i]
        header = list(utt.index.values)
        value = list(utt.values)

        if which_corpus == "atr":
            data_name = f"ATR503_{utt.utt_num}"
        elif which_corpus == "balanced":
            data_name = f"balanced_{utt.utt_num}"
        elif which_corpus == "basic":
            data_name = f"{utt.utt_num}"
    
        with open(str(save_dir / f"{data_name}.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(value)


def main():
    csv_dir = Path("~/lip2sp_pytorch/csv").expanduser()
    atr = csv_dir / "ATR503.csv"
    basic = csv_dir / "BASIC5000.csv"
    balanced = csv_dir / "balanced.csv"
    df_atr = pd.read_csv(str(atr))
    df_basic = pd.read_csv(str(basic))
    df_balanced = pd.read_csv(str(balanced))
    
    data_split_csv_dir = Path("~/dataset/lip/data_split_csv").expanduser()
    train = data_split_csv_dir / "train_all.csv"
    val = data_split_csv_dir / "val.csv"
    test = data_split_csv_dir / "test.csv"
    df_train = pd.read_csv(str(train), header=None)
    df_val = pd.read_csv(str(val), header=None)
    df_test = pd.read_csv(str(test), header=None)

    utt_save_dir = Path("~/dataset/lip/utt").expanduser()
    utt_save_dir_train = utt_save_dir / "train"
    utt_save_dir_val = utt_save_dir / "val"
    utt_save_dir_test = utt_save_dir / "test"
    utt_save_dir_train.mkdir(parents=True, exist_ok=True)
    utt_save_dir_val.mkdir(parents=True, exist_ok=True)
    utt_save_dir_test.mkdir(parents=True, exist_ok=True)

    # print("train")
    # save_utt(df_train, df_atr, df_basic, df_balanced, utt_save_dir_train)

    # print("val")
    # save_utt(df_val, df_atr, df_basic, df_balanced, utt_save_dir_val)

    # print("test")
    # save_utt(df_test, df_atr, df_basic, df_balanced, utt_save_dir_test)

    print("raw")
    utt_raw_save_dir = Path("~/dataset/lip/utt_raw").expanduser()
    utt_raw_save_dir.mkdir(parents=True, exist_ok=True)
    save_utt_raw(df_atr, "atr", utt_raw_save_dir)
    save_utt_raw(df_balanced, "balanced", utt_raw_save_dir)
    save_utt_raw(df_basic, "basic", utt_raw_save_dir)
    


if __name__ == "__main__":
    main()