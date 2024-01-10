from pathlib import Path
import re
from tqdm import tqdm
import pyopenjtalk
import MeCab
import pandas as pd
import demoji
import neologdn
import random
from collections import defaultdict
import hydra
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
from data_process.phoneme_encode import classes2index_tts


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
    
    
def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits, word_freqs):
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


def tokenize(text_parsed, merges):
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
    
    
@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    mecab = CustomMeCabTagger('-r /dev/null -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd')
    data_dir = Path("~/lip2sp_pytorch/csv").expanduser()
    data_path_list = list(data_dir.glob("*.csv"))
    text_list = []
    for data_path in data_path_list:
        df = pd.read_csv(str(data_path))
        text = list(df["text"].values)
        text_list += text
        
    word_freqs = defaultdict(int)
    for text in tqdm(text_list):
        text = text.replace('\n', '').replace('\r', '')
        text = re.sub(r'[“”]', '', text)
        text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = demoji.replace(string=text, repl='')
        text = re.sub(r'[!”#\$%&\’()*+,\-.\/:;?@[\\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠、。,？！｀＋￥％※→←↑↓△▽▷◁▲▼▶◀ゝ…☆]*', '', text)
        text = re.sub('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]', '', text)
        text = neologdn.normalize(text)
        text = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', text)
        n = random.randint(0, 9)
        text = re.sub(f'[0-9]+', str(n), text)
        text = text.lower()
        if len(text) < 30:
            continue
        if re.search(f'[a-z]+', text) is not None:
            continue
        text_parsed = mecab.parseToDataFrame(text)
        text_parsed = text_parsed.loc[~text_parsed['yomi'].isna()]
        text_parsed['phoneme'] = text_parsed['yomi'].apply(lambda x: pyopenjtalk.g2p(x, join=False))
        text_parsed = text_parsed.loc[~(text_parsed['phoneme'] == '')]
        for i, row in text_parsed.iterrows():
            word_freqs[(row["hyousou"], row["yomi"])] += 1
            
    class_to_id, id_to_class = classes2index_tts(cfg)
    last_id = max(class_to_id.values())
    splits = {}
    for hyousou, yomi in word_freqs.keys():
        phoneme = pyopenjtalk.g2p(yomi, join=False)
        splits[(hyousou, yomi)] = phoneme

    vocab_size = 500
    merges = {}
    class_phoneme_dict = {}
    for phoneme in class_to_id.keys():
        class_phoneme_dict[phoneme] = [phoneme]
        
    for i in tqdm(range(vocab_size)):
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits, word_freqs)
        merges[best_pair] = best_pair[0] + best_pair[1]
        class_to_id[best_pair[0] + best_pair[1]] = last_id + 1
        id_to_class[last_id + 1] = best_pair[0] + best_pair[1]
        class_phoneme_dict[best_pair[0] + best_pair[1]] = class_phoneme_dict[best_pair[0]] + class_phoneme_dict[best_pair[1]]
        last_id += 1

    text_parsed_list = []
    for text in tqdm(text_list[:50]):
        text = text.replace('\n', '').replace('\r', '')
        text = re.sub(r'[“”]', '', text)
        text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = demoji.replace(string=text, repl='')
        text = re.sub(r'[!”#\$%&\’()*+,\-.\/:;?@[\\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠、。,？！｀＋￥％※→←↑↓△▽▷◁▲▼▶◀ゝ…☆]*', '', text)
        text = re.sub('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]', '', text)
        text = neologdn.normalize(text)
        text = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', text)
        n = random.randint(0, 9)
        text = re.sub(f'[0-9]+', str(n), text)
        text = text.lower()
        if len(text) < 30:
            continue
        if re.search(f'[a-z]+', text) is not None:
            continue
        text_parsed = mecab.parseToDataFrame(text)
        text_parsed = text_parsed.loc[~text_parsed['yomi'].isna()]
        text_parsed['phoneme'] = text_parsed['yomi'].apply(lambda x: pyopenjtalk.g2p(x, join=False))
        text_parsed = text_parsed.loc[~(text_parsed['phoneme'] == '')]
        text_parsed = tokenize(text_parsed, merges)
        phoneme_list = text_parsed["phoneme"].values
        sub_phoneme_list = text_parsed["sub_phoneme"].values
        sub_phoneme_seq_list = []
        for i in range(len(text_parsed)):
            phoneme = phoneme_list[i]
            sub_phoneme = sub_phoneme_list[i]
            sub_phoneme_index_list = [0 for _ in range(len(phoneme))]
            for j in range(len(sub_phoneme)):
                token = sub_phoneme[j]
                cnt = 0
                for k in range(len(sub_phoneme)):
                    p = class_phoneme_dict["".join(sub_phoneme[k])]
                    if token == sub_phoneme[k]:
                        for l in range(cnt, cnt + len(p)):
                            sub_phoneme_index_list[l] = j
                    cnt += len(p)
            sub_phoneme_seq = []
            for sub_phoneme_index in sub_phoneme_index_list:
                sub_phoneme_seq.append(sub_phoneme[sub_phoneme_index])    
            sub_phoneme_seq_list.append(sub_phoneme_seq)
        text_parsed["sub_phoneme_seq"] = sub_phoneme_seq_list
        text_parsed_list.append(text_parsed)
        print(len(text_parsed_list))
        
        
        
if __name__ == "__main__":
    main()