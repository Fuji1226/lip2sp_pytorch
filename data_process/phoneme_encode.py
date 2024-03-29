"""
音素アラインメントからOneHot表現を取得
この前にcopy_alignment.pyでアラインメントにミスしたデータを取り除いておく必要がある
"""
from pathlib import Path
import numpy as np
import re


IGNORE_INDEX = 0
SOS_INDEX = 1  
EOS_INDEX = 2


def classes2index_tts():
    phonemes = [
        "A",
        "E",
        "I",
        "N",
        "O",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "v",
        "w",
        "y",
        "z",
        "pau",
        "sil",
    ]

    extra_symbols = [
        "^",  # 文の先頭を表す特殊記号 <SOS>
        "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
        "?",  # 文の末尾を表す特殊記号 <EOS> (疑問系)
        "_",  # ポーズ
        "#",  # アクセント句境界
        "[",  # ピッチの上がり位置
        "]",  # ピッチの下がり位置
    ]

    _pad = "~"

    class_list = [_pad] + extra_symbols + phonemes
    class_to_id = {s: i for i, s in enumerate(class_list)}
    id_to_class = {i: s for i, s in enumerate(class_list)}
    return class_to_id, id_to_class


def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    # 未定義 (xx) の場合、コンテキストの取りうる値以外の適当な値
    if match is None:
        return -50
    return int(match.group(1))


def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)

        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        
        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1:
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")

    return PP


def get_keys_from_value(dict, val):
    """
    辞書に対してvalueからkeyを得る
    一度数値列に変換した音素列をもう一度音素列に変換するために使用
    """
    for k, v in dict.items():
        if v == val:
            return k


def main():
    import pyopenjtalk
    text = "あらゆる現実を、すべて自分のほうへねじ曲げたのだ。"
    labels = pyopenjtalk.extract_fullcontext(text)
    PP = pp_symbols(labels)
    class_to_id, id_to_class = classes2index_tts()
    PP = [class_to_id[x] for x in PP]
    breakpoint()
    
    
if __name__ == "__main__":
    main()