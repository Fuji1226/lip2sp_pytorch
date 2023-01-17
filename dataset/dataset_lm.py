import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pyopenjtalk
import pandas as pd
import MeCab
import random

from data_process.phoneme_encode import classes2index_tts, SOS_INDEX, EOS_INDEX
from dataset.utils import get_utt_wiki, get_utt
from dataset.dataset_lipreading import adjust_max_data_len


class DatasetLM(Dataset):
    def __init__(self, data_path, train_data_path, transform, cfg, load_wiki):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.classes_index = classes2index_tts()
        if load_wiki:
            self.path_text_pair_list = get_utt_wiki(data_path, cfg)
        else:
            self.path_text_pair_list = get_utt(data_path)

        print(f"n = {self.__len__()}")

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path, text = self.path_text_pair_list[index]
        speaker = data_path.parents[1].name
        label = data_path.stem

        text = self.transform(text, self.classes_index)

        # 実際の入力はeosを含まないので1引く
        text_len = torch.tensor(text.shape[0] - 1)
        return text, text_len


class TransformLM:
    def __init__(self, cfg, train_val_test):
        self.cfg = cfg
        self.train_val_test = train_val_test
        self.mecab = MeCab.Tagger('-Owakati')

    def get_word_list(self, text):
        """
        mecabによる分かち書きの結果を分割して単語のリストにする
        """
        text = text.split(" ")
        word_list = []
        word = []
        for i in range(len(text)):
            if text[i] == "pau":
                word_list.append(word)
                word = []
            else:
                word.append(text[i])

            if i == len(text) - 1:
                word_list.append(word)

        return word_list

    def text2index(self, text, classes_index):
        """
        音素ラベルを数値列に変換
        get_word_listしていない場合(textにg2pした後のやつに対して適用)
        """
        text = text.split(" ")
        text.insert(0, "sos")
        text.append("eos")
        text = [classes_index[t] if t in classes_index.keys() else None for t in text]
        assert (None in text) is False
        return torch.tensor(text)

    def text2index_word_list(self, word_list, classes_index):
        """
        get_word_listにより得られたword_listに対して任意の単語数分インデックスに変換して取得
        """
        result = []

        if len(word_list) > self.cfg.train.learning_seq_len:
            start_index = random.randint(0, len(word_list) - self.cfg.train.learning_seq_len - 1)
            for word in word_list[start_index:start_index + self.cfg.train.learning_seq_len]:
                word = [classes_index[i] if i in classes_index.keys() else None for i in word]
                result += word
        else:
            for word in word_list:
                word = [classes_index[i] if i in classes_index.keys() else None for i in word]
                result += word

        result.insert(0, SOS_INDEX)
        result.append(EOS_INDEX)
        return torch.tensor(result)

    def __call__(self, text, classes_index):
        text = self.mecab.parse(text)
        text = pyopenjtalk.g2p(text)
        text = self.get_word_list(text)
        text = self.text2index_word_list(text, classes_index)
        return text


def collate_time_adjust_lm(batch, cfg):
    text, text_len = list(zip(*batch))

    text = adjust_max_data_len(text)

    text = torch.stack(text)
    text_len = torch.stack(text_len)

    return text, text_len