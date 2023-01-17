from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_speaker_idx(data_path):
    """
    話者名を数値に変換し,話者IDとする
    複数話者音声合成で必要になります
    """
    print("\nget speaker idx")
    speaker_idx = {}
    idx_set = {
        "F01_kablab" : 0,
        "F02_kablab" : 1,
        "M01_kablab" : 2,
        "M04_kablab" : 3,
        "F01_kablab_fulldata" : 100,
    }
    for path in sorted(data_path):
        speaker = path.parents[1].name
        if speaker in speaker_idx:
            continue
        else:
            speaker_idx[speaker] = idx_set[speaker]
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def get_stat_load_data(train_data_path):
    print("\nget stat")
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []
    feat_add_mean_list = []
    feat_add_var_list = []
    feat_add_len_list = []
    landmark_mean_list = []
    landmark_var_list = []
    landmark_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))

        lip = npz_key['lip']
        feature = npz_key['feature']
        feat_add = npz_key['feat_add']
        landmark = npz_key['landmark']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])

        feat_add_mean_list.append(np.mean(feat_add, axis=0))
        feat_add_var_list.append(np.var(feat_add, axis=0))
        feat_add_len_list.append(feat_add.shape[0])

        landmark_mean_list.append(np.mean(landmark, axis=(0, 2)))
        landmark_var_list.append(np.var(landmark, axis=(0, 2)))
        landmark_len_list.append(landmark.shape[0])
        
    return lip_mean_list, lip_var_list, lip_len_list, \
        feat_mean_list, feat_var_list, feat_len_list, \
            feat_add_mean_list, feat_add_var_list, feat_add_len_list, \
                landmark_mean_list, landmark_var_list, landmark_len_list


def calc_mean_var_std(mean_list, var_list, len_list):
    mean_square_list = list(np.square(mean_list))

    square_mean_list = []
    for var, mean_square in zip(var_list, mean_square_list):
        square_mean_list.append(var + mean_square)

    mean_len_list = []
    square_mean_len_list = []
    for mean, square_mean, len in zip(mean_list, square_mean_list, len_list):
        mean_len_list.append(mean * len)
        square_mean_len_list.append(square_mean * len)

    mean = sum(mean_len_list) / sum(len_list)
    var = sum(square_mean_len_list) / sum(len_list) - mean ** 2
    std = np.sqrt(var)
    return mean, var, std


def get_utt(data_path):
    print("--- get utterance ---")
    text_dir = Path("~/dataset/lip/utt").expanduser()

    path_text_pair_list = []
    for path in tqdm(data_path):
        text_path = text_dir / path.parents[2].name / path.parents[1].name / f"{path.stem}.csv"
        df = pd.read_csv(str(text_path))
        text = df.pronounce.values[0]
        path_text_pair_list.append([path, text])

    return path_text_pair_list


def get_utt_wiki(data_path, cfg):
    print("--- get utterance with wikipedia ---")
    text_dir = Path("~/dataset/lip/utt").expanduser()

    path_text_pair_list = []
    print("load atr, balanced, basic")
    for path in tqdm(data_path):
        text_path = text_dir / path.parents[2].name / path.parents[1].name / f"{path.stem}.csv"
        df = pd.read_csv(str(text_path))
        text = df.pronounce.values[0]
        path_text_pair_list.append([path, text])

    print("load wikipedia")
    wiki_data = pd.read_csv(str(cfg.train.wiki_path))
    for i in tqdm(range(len(wiki_data))):
        text = wiki_data.iloc[i].pronounce
        path_text_pair_list.append([cfg.train.wiki_path, text])
        
        if cfg.train.debug:
            if len(path_text_pair_list) > 10000:
                break

    return path_text_pair_list


def get_spk_emb(cfg):
    data_dir = Path("~/dataset/lip/emb").expanduser()
    spk_emb_dict = {}

    for speaker in cfg.train.speaker:
        data_path = data_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))

        spk_emb_dict[speaker] = emb

    return spk_emb_dict