from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import joblib
from functools import partial
import torch
import pickle

from data_process.transform import load_data_lrs2, load_data

text_dir = Path("~/dataset/lip/utt").expanduser()
emb_dir = Path("~/dataset/lip/emb").expanduser()
emb_lrs2_dir_pretrain = Path("~/lrs2/emb/pretrain").expanduser()
emb_lrs2_dir_main = Path("~/lrs2/emb/main").expanduser()
data_info_df_main_path = Path("~/lrs2/data_info_main.csv").expanduser()
data_info_df_pretrain_path = Path("~/lrs2/data_info_pretrain.csv").expanduser()
emb_lip2wav_dir = Path("~/Lip2Wav/emb").expanduser()


def select_data(data_root, data_bbox_root, data_landmark_root, data_df, cfg):
    print(f"\nselect existing data")
    data_path_list = []
    for speaker in cfg.train.speaker:
        for i in tqdm(range(len(data_df))):
            filename = data_df.iloc[i].values[0]
            for corpus in cfg.train.corpus:
                if re.search(corpus, str(filename)):
                    video_path = data_root / speaker / f"{filename}.mp4"
                    audio_path = data_root / speaker / f"{filename}.wav"
                    bbox_path = data_bbox_root / speaker / f"{filename}.csv"
                    landmark_path = data_landmark_root / speaker / f"{filename}.csv"
                    text_path = text_dir / f"{filename}.csv"
                    if video_path.exists() and audio_path.exists() and bbox_path.exists() and landmark_path.exists() and text_path.exists():
                        data_path_list.append([video_path, audio_path, bbox_path, landmark_path, text_path])
                        
    return data_path_list


def select_data_lrs2(data_root, data_bbox_root, data_landmark_root, data_df, cfg, which_data):
    print(f"\nselect existing data")
    data_df = data_df.rename(columns={0: "filename"})
    data_df["id"] = data_df["filename"].apply(lambda x: str(x.split("/")[0]))
    
    # 学習データが多すぎるので削る
    if which_data == "train":
        if data_root.stem == "main":
            info_df = pd.read_csv(str(data_info_df_main_path))
        elif data_root.stem == "pretrain":
            info_df = pd.read_csv(str(data_info_df_pretrain_path))
        info_df = info_df.sort_values("n_data")
        info_df = info_df.tail(cfg.train.lrs2_n_train_speaker_used)
        info_df["id"] = info_df["id"].astype(str)
        data_df = data_df.loc[data_df["id"].isin(info_df["id"].to_list())]
        
    if cfg.train.debug:
        data_df = data_df.iloc[:1000]
    
    data_path_list = []
    for i in tqdm(range(len(data_df))):
        filename = data_df.iloc[i]["filename"]
        video_path = data_root / f"{filename}.mp4"
        bbox_path = data_bbox_root / f"{filename}.csv"
        landmark_path = data_landmark_root / f"{filename}.csv"
        if video_path.exists() and bbox_path.exists() and landmark_path.exists():
            data_path_list.append([video_path, bbox_path, landmark_path])

    return data_path_list


def get_speaker_idx(data_path):
    print("\nget speaker idx")
    speaker_idx = {}
    idx_dict = {
        "F01_kablab" : 0,
        "F02_kablab" : 1,
        "M01_kablab" : 2,
        "M04_kablab" : 3,
        "F01_kablab_all" : 4,
    }
    for path in data_path:
        speaker = path.parents[1].name
        if speaker in speaker_idx:
            continue
        else:
            if speaker in idx_dict:
                speaker_idx[speaker] = idx_dict[speaker]
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def get_speaker_idx_lrs2(data_path, cfg):
    print(f"\nget speaker idx lrs2")
    speaker_idx = {}
    if cfg.train.which_external_data == "lrs2_main":
        train_df_path = Path("~/lrs2/train.txt")
    elif cfg.train.which_external_data == "lrs2_pretrain":
        train_df_path = Path("~/lrs2/pretrain.txt")
        
    train_data_df = pd.read_csv(str(train_df_path), header=None)
    train_data_df = train_data_df.rename(columns={0: "filename_all"})
    train_data_df["id"] = train_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
    idx_list = train_data_df["id"].unique()
    idx_dict = dict([[x, i + 100000] for i, x in enumerate(idx_list)])
    for path in data_path:
        speaker = path.parents[1].name
        if speaker in speaker_idx:
            continue
        else:
            if speaker in idx_dict:
                speaker_idx[speaker] = idx_dict[speaker]
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def get_speaker_idx_lip2wav(data_path):
    print(f"\nget speaker idx lip2wav")
    speaker_idx = {}
    idx_dict = {
        "chem" : 200000,
        "chess" : 200001,
        "dl" : 200002,
        "eh" : 200003,
        "hs" : 200004,
    }
    for path in data_path:
        speaker = path.parents[1].name
        if speaker in speaker_idx:
            continue
        else:
            if speaker in idx_dict:
                speaker_idx[speaker] = idx_dict[speaker]
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def get_speaker_idx_jsut():
    print(f"\nget speaker idx jsut")
    speaker_idx = {"female": 300000}
    print(f"speaker_idx = {speaker_idx}")
    return speaker_idx


def get_speaker_idx_jvs():
    print(f"\nget speaker idx jvs")
    speaker_idx = dict([[f"jvs{i:03d}", 400000 + i] for i in range(1, 101)])
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
        # feat_add = npz_key['feat_add']
        # landmark = npz_key['landmark']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])

        # feat_add_mean_list.append(np.mean(feat_add, axis=0))
        # feat_add_var_list.append(np.var(feat_add, axis=0))
        # feat_add_len_list.append(feat_add.shape[0])

        # landmark_mean_list.append(np.mean(landmark, axis=(0, 2)))
        # landmark_var_list.append(np.var(landmark, axis=(0, 2)))
        # landmark_len_list.append(landmark.shape[0])
        
    return lip_mean_list, lip_var_list, lip_len_list, \
        feat_mean_list, feat_var_list, feat_len_list, \
            feat_add_mean_list, feat_add_var_list, feat_add_len_list, \
                landmark_mean_list, landmark_var_list, landmark_len_list
                
                
def get_stat_load_data_lip_face(train_data_path):
    print("\nget stat")
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    face_mean_list = []
    face_var_list = []
    face_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []
    feat_add_mean_list = []
    feat_add_var_list = []
    feat_add_len_list = []

    for path in tqdm(train_data_path):
        npz_key = np.load(str(path))

        lip = npz_key['lip']
        face = npz_key['face']
        feature = npz_key['feature']
        feat_add = npz_key['feat_add']

        lip_mean_list.append(np.mean(lip, axis=(1, 2, 3)))
        lip_var_list.append(np.var(lip, axis=(1, 2, 3)))
        lip_len_list.append(lip.shape[-1])
        
        face_mean_list.append(np.mean(face, axis=(1, 2, 3)))
        face_var_list.append(np.var(face, axis=(1, 2, 3)))
        face_len_list.append(face.shape[-1])

        feat_mean_list.append(np.mean(feature, axis=0))
        feat_var_list.append(np.var(feature, axis=0))
        feat_len_list.append(feature.shape[0])

        feat_add_mean_list.append(np.mean(feat_add, axis=0))
        feat_add_var_list.append(np.var(feat_add, axis=0))
        feat_add_len_list.append(feat_add.shape[0])
        
    return lip_mean_list, lip_var_list, lip_len_list, \
        face_mean_list, face_var_list, face_len_list, \
            feat_mean_list, feat_var_list, feat_len_list, \
                feat_add_mean_list, feat_add_var_list, feat_add_len_list


def load_and_calc_mean_var(video_path, audio_path, bbox_path, landmark_path, text_path, cfg, aligner):
    wav, lip, feature, data_len, text = load_data(video_path, audio_path, bbox_path, landmark_path, text_path, cfg, aligner)
    lip_mean = np.mean(lip, axis=(1, 2, 3))
    lip_var = np.var(lip, axis=(1, 2, 3))
    feat_mean = np.mean(feature, axis=0)
    feat_var = np.var(feature, axis=0)
    lip_len = lip.shape[-1]
    feat_len = feature.shape[0]
    return lip_mean, lip_var, lip_len, feat_mean, feat_var, feat_len


def get_stat_load_data_raw(data_path_list, cfg, aligner):
    print(f"\nget stat")
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []

    print("multi processing")
    res = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(partial(load_and_calc_mean_var, cfg=cfg, aligner=aligner))(
            video_path, audio_path, bbox_path, landmark_path, text_path
        ) for video_path, audio_path, bbox_path, landmark_path, text_path in tqdm(data_path_list)
    )
    for lip_mean, lip_var, lip_len, feat_mean, feat_var, feat_len in res:
        lip_mean_list.append(lip_mean)
        lip_var_list.append(lip_var)
        lip_len_list.append(lip_len)
        feat_mean_list.append(feat_mean)
        feat_var_list.append(feat_var)
        feat_len_list.append(feat_len)

    return lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list


def load_and_calc_mean_var_lrs2(video_path, bbox_path, landmark_path, cfg, aligner):
    wav, lip, feature, data_len = load_data_lrs2(video_path, bbox_path, landmark_path, cfg, aligner)
    lip_mean = np.mean(lip, axis=(1, 2, 3))
    lip_var = np.var(lip, axis=(1, 2, 3))
    feat_mean = np.mean(feature, axis=0)
    feat_var = np.var(feature, axis=0)
    lip_len = lip.shape[-1]
    feat_len = feature.shape[0]
    return lip_mean, lip_var, lip_len, feat_mean, feat_var, feat_len


def get_stat_load_data_lrs2(data_path_list, cfg, aligner):
    print(f"\nget stat")
    lip_mean_list = []
    lip_var_list = []
    lip_len_list = []
    feat_mean_list = []
    feat_var_list = []
    feat_len_list = []

    print("multi processing")
    res = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(partial(load_and_calc_mean_var_lrs2, cfg=cfg, aligner=aligner))(
            video_path, bbox_path, landmark_path
        ) for video_path, bbox_path, landmark_path in tqdm(data_path_list)
    )
    for lip_mean, lip_var, lip_len, feat_mean, feat_var, feat_len in res:
        lip_mean_list.append(lip_mean)
        lip_var_list.append(lip_var)
        lip_len_list.append(lip_len)
        feat_mean_list.append(feat_mean)
        feat_var_list.append(feat_var)
        feat_len_list.append(feat_len)

    return lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list


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


def get_recorded_synth_label(path):
    if ("ATR" in path.stem) or ("balanced" in path.stem):
        label = 1
    elif "BASIC5000" in path.stem:
        if int(str(path.stem).split("_")[1]) > 2500:
            label = 0
        else:
            label = 1
    else:
        label = 0
    return label


def get_utt_label(data_path):
    print("--- get utterance ---")
    path_text_label_list = []
    for path in tqdm(data_path):
        text_path = text_dir / f"{path.stem}.txt"
        df = pd.read_csv(str(text_path), header=None)
        text = df[0].values[0]
        label = get_recorded_synth_label(path)
        path_text_label_list.append([path, text, label])

    return path_text_label_list


def get_utt_wiki(data_path, cfg):
    print("--- get utterance with wikipedia ---")

    path_text_pair_list = []
    print("load atr, balanced, basic")
    for path in tqdm(data_path):
        text_path = text_dir / path.parents[2].name / f"{path.stem}.csv"
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
    spk_emb_dict = {}
    for speaker in cfg.train.speaker:
        data_path = emb_dir / speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict


def get_spk_emb_lrs2():
    spk_emb_dict = {}
    speaker_list = list(emb_lrs2_dir_pretrain.glob("*"))
    for speaker in speaker_list:
        data_path = speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker.stem] = emb

    # validationの話者はpretrainに含まれないので別で取得
    speaker_list = list(emb_lrs2_dir_main.glob("*"))
    for speaker in speaker_list:
        if speaker in spk_emb_dict:
            continue
        data_path = speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker.stem] = emb
        
    return spk_emb_dict


def get_spk_emb_lip2wav():
    spk_emb_dict = {}
    speaker_list = list(emb_lip2wav_dir.glob("*"))
    for speaker in speaker_list:
        data_path = speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker.stem] = emb
    return spk_emb_dict


def get_spk_emb_jsut():
    data_path = Path("~/dataset/jsut_ver1.1/emb.npy").expanduser()
    emb = np.load(str(data_path))
    emb = emb / np.linalg.norm(emb)
    spk_emb_dict = {"female": emb}
    return spk_emb_dict


def get_spk_emb_jvs():
    data_path = Path("~/dataset/jvs_ver1/emb").expanduser()
    speaker_list = [f"jvs{i:03d}" for i in range(1, 101)]
    spk_emb_dict = {}
    for speaker in speaker_list:
        emb = np.load(str(data_path / speaker / "emb.npy"))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker] = emb
    return spk_emb_dict


def adjust_max_data_len(data):
    """
    minibatchの中で最大のdata_lenに合わせて0パディングする
    data : (..., T)
    """
    max_data_len = 0
    max_data_len_id = 0

    # minibatchの中でのdata_lenの最大値と，そのデータのインデックスを取得
    for idx, d in enumerate(data):
        if max_data_len < d.shape[-1]:
            max_data_len = d.shape[-1]
            max_data_len_id = idx

    new_data = []

    # data_lenが最大のデータに合わせて0パディング
    for d in data:
        d_padded = torch.zeros_like(data[max_data_len_id])

        for t in range(d.shape[-1]):
            d_padded[..., t] = d[..., t]

        new_data.append(d_padded)
    
    return new_data