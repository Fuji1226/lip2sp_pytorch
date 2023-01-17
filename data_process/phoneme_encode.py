"""
音素アラインメントからOneHot表現を取得
この前にcopy_alignment.pyでアラインメントにミスしたデータを取り除いておく必要がある
"""
from pathlib import Path
import numpy as np


IGNORE_INDEX = 0
SOS_INDEX = 1  
EOS_INDEX = 2


def get_alignment(data_root):
    """
    音素アラインメントを行った.labファイルまでのパスを取得
    """
    data_path = list(data_root.glob("*.lab"))
    assert data_path is not None
    return data_path


def get_classes(data_path):
    """
    データに含まれる音素全種類を取得
    音素ラベルをOnehot表現に変換するために必要
    """
    phoneme = []

    for [_, alignment_path] in data_path:
        with open(str(alignment_path), "r") as f:
            data = f.read()

            # 改行を空白に変換してから，空白で分割する
        data = data.replace("\n", " ")
        data = data.split(" ")
        
        # 音素ラベルを取得
        phoneme.append(data[2::3])
    
    classes = []

    # マスク用のtokenを追加
    classes.append("mask")  # mask_id = 0

    # 開始を表す"sos"を追加
    # classes.append("sos")   # sos_id = 1
    classes.append("silB")   # sos_id = 1

    # 終了を表す"eos"を追加
    # classes.append("eos")   # eos_id = 2
    classes.append("silE")   # eos_id = 2

    # データに含まれる音素から，音素ラベル全種類を取得
    for each_p in phoneme:
        for p in each_p:
            if p in classes:
                continue
            else:
                classes.append(p)

    print("\n--- label check ---")
    print(f"n_classes = {len(classes)}")

    for idx, label in enumerate(classes):
        print(f"{idx} : {label}")
    print("")
    return classes


def get_classes_ctc(data_path):
    phoneme = []

    for [_, alignment_path] in data_path:
        with open(alignment_path, "r") as f:
            data = f.read()

            # 改行を空白に変換してから，空白で分割する
        data = data.replace("\n", " ")
        data = data.split(" ")
        
        # 音素ラベルを取得
        phoneme.append(data[2::3])
    
    classes = []

    # blank index
    classes.append("-")  # mask_id = 0

    # データに含まれる音素から，音素ラベル全種類を取得
    for each_p in phoneme:
        for p in each_p:
            if p in classes:
                continue
            else:
                classes.append(p)

    print("\n--- label check ---")
    print(f"n_classes = {len(classes)}")

    for idx, label in enumerate(classes):
        print(f"{idx} : {label}")
    print("")
    return classes


def classes2index(classes):
    """
    音素の種類であるclassesを数字に変換
    音素列を数値化するために使用する
    """
    classes_index = {}
    
    for i, c in enumerate(classes):
        classes_index[c] = i

    classes_index = {
        "mask" : 0,
        "silB" : 1,
        "silE" : 2,
        "u" : 3,
        "s" : 4,
        "g" : 5,
        "i" : 6,
        "t" : 7,
        "a" : 8,
        "n" : 9,
        "sp" : 10,
        "m" : 11,
        "r" : 12,
        "e" : 13,
        "k" : 14,
        "o" : 15,
        "ky" : 16,
        "o:" : 17,
        "ch" : 18,
        "d" : 19,
        "q" : 20,
        "w" : 21,
        "f" : 22,
        "ts" : 23,
        "p" : 24,
        "N" : 25,
        "sh" : 26,
        "h" : 27,
        "y" : 28,
        "z" : 29,
        "i:" : 30,
        "b" : 31,
        "u:" : 32,
        "ny" : 33,
        "e:" : 34,
        "ry" : 35,
        "a:" : 36,
        "j" : 37,
        "gy" : 38,
        "by" : 39,
        "hy" : 40,
        "py" : 41,
        "my" : 42,
        "dy" : 43,
    }

    return classes_index


def classes2index_tts():
    classes_index = {
        "mask" : IGNORE_INDEX,
        "sos" : SOS_INDEX,
        "eos" : EOS_INDEX,
        "a" : 3,
        "i" : 4,
        "u" : 5,
        "e" : 6,
        "o" : 7,
        "A" : 8,
        "I" : 9,
        "U" : 10,
        "E" : 11,
        "O" : 12,
        "k" : 13,
        "g" : 14,
        "s" : 15,
        "z" : 16,
        "t" : 17,
        "ts" : 18,
        "d" : 19,
        "n" : 20,
        "h" : 21,
        "f" : 22,
        "b" : 23,
        "p" : 24,
        "m" : 25,
        "y" : 26,
        "r" : 27,
        "w" : 28,
        "v" : 29,
        "ky" : 30,
        "gy" : 31,
        "sh" : 32,
        "j" : 33,
        "ch" : 34,
        "ty" : 35,
        "dy" : 36,
        "ny" : 37,
        "hy" : 38,
        "by" : 39,
        "py" : 40,
        "my" : 41,
        "ry" : 42,
        "N" : 43,
        "cl" : 44,
        "pau" : 45,
        "sil" : 46,
    }
    return classes_index


def get_phoneme_info(alignment_path):
    """
    音素ラベルとその継続時間を取得

    phoneme : 音素アラインメントされた結果から得られる音素列
    duration : 各音素の継続時間。[start_time, end_time]
    """
    duration = []

    with open(alignment_path, "r") as f:
        data = f.read()
    
    # 改行を空白に変換してから，空白で分割する
    data = data.replace("\n", " ")
    data = data.split(" ")
    
    # 音素ラベルを取得
    phoneme = data[2::3]

    # 各音素ラベルの開始時刻，終了時刻を取得
    start_time = data[0::3]
    end_time = data[1::3]

    for i in range(0, len(data), 3):
        duration.append(data[i:i+2])

    duration = duration[:-1]
    return phoneme, duration


def get_keys_from_value(dict, val):
    """
    辞書に対してvalueからkeyを得る
    一度数値列に変換した音素列をもう一度音素列に変換するために使用
    """
    for k, v in dict.items():
        if v == val:
            return k


def main():
    data_root = Path("~/dataset/alignment/train/F01_kablab").expanduser()
    data_path = get_alignment(data_root)

    idx = np.random.randint(0, 100, (1)).item()
    data_path_exp = data_path[idx]
    phoneme, duration = get_phoneme_info(str(data_path_exp))

    dict_exp = {
        "a" : 1,
        "i" : 2,
    }
    sample_phoneme = ["a", "i"]

    phoneme_index = [dict_exp[i] if i in dict_exp.keys() else None for i in sample_phoneme]

    phoneme_recon = []
    for i in phoneme_index:
        phoneme_recon.append(get_keys_from_value(dict_exp, i))


    phoneme_length = 20
    print(len(phoneme))
    print(len(duration))
    
    if len(phoneme) > phoneme_length:
        idx = np.random.randint(0, len(phoneme) - phoneme_length - 1, (1,)).item()
        breakpoint()
        phoneme = phoneme[idx:idx + phoneme_length]
        duration = duration[idx:idx + phoneme_length]
    breakpoint()


if __name__ == "__main__":
    main()
