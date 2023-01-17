import re
from itertools import chain
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import pykakasi
import csv

debug = False
debug_iter = 3

data_path = Path("~/dataset/wiki/datafiles/sentences/text.json").expanduser()
checker = re.compile("^[一-龥々ヶぁ-んァ-ヴー、]+[ぁ-ん]+[一-龥々ヶぁ-んァ-ヴー、]+。$").match
middle_checker = re.compile("^[一-龥々ヶぁ-んァ-ヴー、。]+$").match
kakasi = pykakasi.kakasi()
seq_len_limit_upper = 130   # atr, balanced, basicの中での最大系列長を参考に設定

header = ["text", "pronounce"]
save_dir_limit = Path(f"~/dataset/wiki/datafiles/sentences/limit_{seq_len_limit_upper}").expanduser()
save_dir_no_limit = Path(f"~/dataset/wiki/datafiles/sentences/no_limit").expanduser()
save_dir_limit.mkdir(parents=True, exist_ok=True)
save_dir_no_limit.mkdir(parents=True, exist_ok=True)


def process(idx, data):
    text = data.text
    selected_list = []
    for txt in text.split("\n"):
        if '。' not in txt:
            continue

        for t in txt.replace("。", "。__dummy__").split("__dummy__"):
            if not t.split():
                continue
            if t[-1] != '。':
                continue
            if "、。" in "".join(t.split(" ")):
                continue
            if "、、" in "".join(t.split(" ")):
                continue

            if checker(t):
                pass
            elif '、' in t:
                t_list = []
                for part in t.split("、")[::-1]:
                    if middle_checker(part):
                        t_list.append(part)
                    else:
                        break
                if t_list:
                    t = "、".join(t_list[::-1])
                    if not checker(t):
                        continue
                else:
                    continue
            else:
                continue

            selected_list.append(t)
    
    return selected_list


def pronounce(selected_list):
    kanji_yomi_list = []
    kanji_yomi_list_no_limit = []
    for selected_data in selected_list:
        for data in selected_data:
            yomi_list = kakasi.convert(data)
            yomi_hira_list = [yomi["hira"] for yomi in yomi_list]
            yomi_hira = "".join(yomi_hira_list)

            if len(yomi_hira) < seq_len_limit_upper:
                kanji_yomi_list.append([data, yomi_hira])
            
            kanji_yomi_list_no_limit.append([data, yomi_hira])

    return kanji_yomi_list, kanji_yomi_list_no_limit


def main():
    with open(str(save_dir_limit / "text.csv"), "a") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    with open(str(save_dir_no_limit / "text.csv"), "a") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    with data_path.open('r') as file:
        num_lines = sum(1 for _ in file)

    pbar = tqdm(total=num_lines)
    iter_cnt = 0
    for df in pd.read_json(data_path, lines=True, chunksize=100):
        length = len(df)

        selected_list = Parallel(n_jobs=-1)(
            [delayed(process)(idx, data) for idx, data in df.iterrows()]
        )
        if len(selected_list) == 0:
            continue

        kanji_yomi_list, kanji_yomi_list_no_limit = pronounce(selected_list)
        # print(len(kanji_yomi_list), len(kanji_yomi_list_no_limit), len(kanji_yomi_list) / len(kanji_yomi_list_no_limit))

        with open(str(save_dir_limit / "text.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerows(kanji_yomi_list)

        with open(str(save_dir_no_limit / "text.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerows(kanji_yomi_list_no_limit)

        pbar.update(length)
        iter_cnt += 1
        if debug:
            if iter_cnt > debug_iter:
                break


def check_length():
    data_dir = Path("~/lip2sp_pytorch/csv").expanduser()
    atr = data_dir / "ATR503.csv"
    balanced = data_dir / "balanced.csv"
    basic = data_dir / "BASIC5000.csv"
    csv_path_list = [atr, balanced, basic]
    max_seq_len = 0

    for path in csv_path_list:
        df = pd.read_csv(str(path))
        
        for i in range(len(df)):
            text = df.iloc[i].pronounce
            if max_seq_len < len(text):
                max_seq_len = len(text)
                print(max_seq_len, path, i, text)

    print(max_seq_len)


def load_wiki():
    save_dir_limit = Path(f"~/dataset/wiki/datafiles/sentences/limit_{seq_len_limit_upper}/text.csv").expanduser()
    wiki_data = pd.read_csv(str(save_dir_limit))


if __name__ == "__main__":
    # main()
    # check_length()
    load_wiki()