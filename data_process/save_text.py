from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pyopenjtalk
import csv
import shutil
from phoneme_encode import classes2index_tts, pp_symbols


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


def save_utt_jsut():
    jsut_dir = Path("~/dataset/jsut_ver1.1").expanduser()
    corpus_list = [
        "basic5000", "countersuffix26", "loanword128", "onomatopee300", "precedent130", 
        "travel1000", "utparaphrase512", "voiceactress100"
    ]
    for corpus in corpus_list:
        corpus_dir = jsut_dir / corpus
        data_path = corpus_dir.glob("*.")
        
        
def get_text():
    text_dir = Path("~/dataset/jsut_ver1.1").expanduser()
    corpus_list = [
        "basic5000", "countersuffix26", "loanword128", "onomatopee300", "precedent130", 
        "travel1000", "utparaphrase512", "voiceactress100"
    ]
    df_list = []
    
    for corpus in corpus_list:
        corpus_text_path = text_dir / corpus / "transcript_utf8.txt"
        df = pd.read_csv(str(corpus_text_path), header=None)
        df_fix = df.copy()
        df_fix["filename"] = df[0].apply(lambda x : str(x.split(":")[0]))
        df_fix["text"] = df[0].apply(lambda x : str(x.split(":")[1]))
        df_fix = df_fix.drop(columns=[0])
        df_fix = df_fix.reset_index(drop=True)
        if corpus == "basic5000":
            df_fix = df_fix[2500:]
        df_list.append(df_fix)
        
    df = pd.concat(df_list)
    # class_to_id, id_to_class = classes2index_tts()
    # df["text"] = df["text"].apply(functools.partial(preprocess, class_to_id=class_to_id))
    return df


def preprocess(text, class_to_id):
    text = pyopenjtalk.extract_fullcontext(text)
    text = pp_symbols(text)
    text = [class_to_id[t] for t in text]
    return text


def main():
    # csv_dir = Path("~/lip2sp_pytorch/csv").expanduser()
    # atr = csv_dir / "ATR503.csv"
    # basic = csv_dir / "BASIC5000.csv"
    # balanced = csv_dir / "balanced.csv"
    # df_atr = pd.read_csv(str(atr))
    # df_basic = pd.read_csv(str(basic))
    # df_balanced = pd.read_csv(str(balanced))
    
    # data_split_csv_dir = Path("~/dataset/lip/data_split_csv").expanduser()
    # train = data_split_csv_dir / "train_all.csv"
    # val = data_split_csv_dir / "val.csv"
    # test = data_split_csv_dir / "test.csv"
    # df_train = pd.read_csv(str(train), header=None)
    # df_val = pd.read_csv(str(val), header=None)
    # df_test = pd.read_csv(str(test), header=None)

    # utt_save_dir = Path("~/dataset/lip/utt").expanduser()
    # utt_save_dir_train = utt_save_dir / "train"
    # utt_save_dir_val = utt_save_dir / "val"
    # utt_save_dir_test = utt_save_dir / "test"
    # utt_save_dir_train.mkdir(parents=True, exist_ok=True)
    # utt_save_dir_val.mkdir(parents=True, exist_ok=True)
    # utt_save_dir_test.mkdir(parents=True, exist_ok=True)

    # print("train")
    # save_utt(df_train, df_atr, df_basic, df_balanced, utt_save_dir_train)

    # print("val")
    # save_utt(df_val, df_atr, df_basic, df_balanced, utt_save_dir_val)

    # print("test")
    # save_utt(df_test, df_atr, df_basic, df_balanced, utt_save_dir_test)

    # print("raw")
    # utt_raw_save_dir = Path("~/dataset/lip/utt_raw").expanduser()
    # utt_raw_save_dir.mkdir(parents=True, exist_ok=True)
    # save_utt_raw(df_atr, "atr", utt_raw_save_dir)
    # save_utt_raw(df_balanced, "balanced", utt_raw_save_dir)
    # save_utt_raw(df_basic, "basic", utt_raw_save_dir)
    
    df = get_text()
    class_to_id, id_to_class = classes2index_tts()
    text_dir = Path("~/dataset/lip/utt_test").expanduser()
    text_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(df.shape[0])):
        data = df.iloc[i]
        filename = data["filename"]
        text = data["text"]
        with open(str(text_dir / f"{filename}.txt"), "w") as f:
            f.write(text)
        break
    
    text_path = list(text_dir.glob("*.txt"))
    for path in text_path:
        text = pd.read_csv(str(path), header=None).iloc[0].values[0]
        print(text)
        text = preprocess(text, class_to_id)
        print(text)


if __name__ == "__main__":
    main()