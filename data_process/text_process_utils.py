import MeCab
import pandas as pd
from typing import Dict, Tuple
from collections import defaultdict


class CustomMeCabTagger(MeCab.Tagger):

    COLUMNS = ['hyousou', 'hinnshi', 'hinnshi_sai1', 'hinnshi_sai2', 'hinnshi_sai3', 'katuyou_kata', 'katuyou_kei', 'gennkeki', 'yomi', 'hatuonn']

    def parseToDataFrame(self, text: str) -> pd.DataFrame:
        """テキストを parse した結果を Pandas DataFrame として返す"""
        results = []
        for line in self.parse(text).split('\n'):
            if line == 'EOS':
                break
            surface, feature = line.split('\t')
            feature = [None if f == '*' else f for f in feature.split(',')]
            results.append([surface, *feature])
        return pd.DataFrame(results, columns=type(self).COLUMNS)
    
    

    
    
def compute_pair_freqs(
    splits: Dict[Tuple[str, str], list],
    word_freqs: Dict[Tuple[str, str], int],
) -> Dict[Tuple[str, str], int]:
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(
    a: int,
    b: int,
    splits: Dict[Tuple[str, str], list],
    word_freqs: Dict[Tuple[str, str], int],
) -> Dict[Tuple[str, str], list]:
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


def tokenize(
    text_parsed: pd.DataFrame,
    merges: Dict[str, str],
) -> pd.DataFrame:
    text_parsed = text_parsed.reset_index(drop=True)
    text_parsed["sub_phoneme"] = text_parsed["phoneme"]
    for pair, merge in merges.items():
        for idx, row in text_parsed.iterrows():
            i = 0
            sub_phoneme = row["sub_phoneme"]
            while i < len(sub_phoneme) - 1:
                if sub_phoneme[i] == pair[0] and sub_phoneme[i + 1] == pair[1]:
                    sub_phoneme = sub_phoneme[:i] + [merge] + sub_phoneme[i + 2:]
                else:
                    i += 1
            row["sub_phoneme"] = sub_phoneme
            text_parsed.iloc[idx] = row
    return text_parsed