from pathlib import Path
import numpy as np
import hydra
from tqdm import tqdm
import os
import random
import re
from datetime import datetime
import pandas as pd


text_dir = Path("~/dataset/lip/utt").expanduser()
emb_dir = Path("~/dataset/lip/emb").expanduser()
emb_lrs2_dir_pretrain = Path("~/lrs2/emb/pretrain").expanduser()
emb_lrs2_dir_main = Path("~/lrs2/emb/main").expanduser()
data_info_df_main_path = Path("~/lrs2/data_info_main.csv").expanduser()
data_info_df_pretrain_path = Path("~/lrs2/data_info_pretrain.csv").expanduser()
emb_lip2wav_dir = Path("~/Lip2Wav/emb").expanduser()


def get_datasets(data_root, cfg):
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.train.speaker:
        print(f"{speaker}")
        spk_path_list = []
        spk_path = data_root / speaker / cfg.model.name

        for corpus in cfg.train.corpus:
            spk_path_co = [p for p in spk_path.glob("*.npz") if re.search(f"{corpus}", str(p))]
            if len(spk_path_co) > 1:
                print(f"load {corpus}")
            spk_path_list += spk_path_co
        items += random.sample(spk_path_list, len(spk_path_list))
    return items


def get_datasets_external_data(cfg):
    if cfg.train.which_external_data == "lrs2_main":
        print(f"\n--- get datasets lrs2_main ---")
        data_dir = Path(cfg.train.lrs2_npz_path).expanduser()
    elif cfg.train.which_external_data == "lrs2_pretrain":
        print(f"\n--- get datasets lrs2_pretrain ---")
        data_dir = Path(cfg.train.lrs2_pretrain_npz_path).expanduser()
    elif cfg.train.which_external_data == "lip2wav":
        print(f"\n--- get datasets lip2wav ---")
        data_dir = Path(cfg.train.lip2wav_npz_path).expanduser()
        
    spk_path_list = list(data_dir.glob("*"))
    items = []
    for spk_path in spk_path_list:
        spk_path = spk_path / cfg.model.name
        data_path_list = list(spk_path.glob("*.npz"))
        items += data_path_list
    return items


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


def get_speaker_idx_lrs2(data_path):
    print(f"\nget speaker idx lrs2")
    speaker_idx = {}
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
    for speaker in tqdm(speaker_list):
        data_path = speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker.stem] = emb

    # validationの話者はpretrainに含まれないので別で取得
    speaker_list = list(emb_lrs2_dir_main.glob("*"))
    for speaker in tqdm(speaker_list):
        if speaker in spk_emb_dict:
            continue
        data_path = speaker / "emb.npy"
        emb = np.load(str(data_path))
        emb = emb / np.linalg.norm(emb)
        spk_emb_dict[speaker.stem] = emb
        
    return spk_emb_dict


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    train_data_root = cfg.train.face_cropped_max_size_fps25_train
    train_data_root = Path(train_data_root).expanduser()
    train_data_path = get_datasets(train_data_root, cfg)
    external_data_path = get_datasets_external_data(cfg)
    data_path = train_data_path + external_data_path
    
    speaker_idx_kablab = get_speaker_idx(data_path)
    embs_kablab = get_spk_emb(cfg)
    if cfg.train.which_external_data == "lrs2_main" or cfg.train.which_external_data == "lrs2_pretrain":
        speaker_idx_ex = get_speaker_idx_lrs2(data_path)
        embs_ex = get_spk_emb_lrs2()
        
    
        
    breakpoint()
    
    
if __name__ == "__main__":
    main()