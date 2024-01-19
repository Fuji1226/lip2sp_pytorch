import re
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import random
from functools import partial
from librosa.display import specshow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import wer
import seaborn as sns
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict, defaultdict
import pandas as pd
import copy
from dataset.dataset_npz import KablabDataset, KablabTransform, collate_time_adjust
from dataset.dataset_npz_with_ex import DatasetWithExternalData, TransformWithExternalData, collate_time_adjust_with_external_data
from dataset.dataset import DatasetWithExternalDataRaw, TransformWithExternalDataRaw
from dataset.dataset_bart import DatasetBART, TransformBART, collate_time_adjust_bart
from data_process.feature import wav2mel
from data_process.phoneme_encode import get_keys_from_value
from model.raven import E2E as MyRAVEN
from model.vatlm import MyVATLM
from model.avhubert import MyAVHubertModel


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def set_config(cfg):
    if cfg.train.debug:
        cfg.train.max_epoch = 2

    if cfg.model.fps == 25:
        cfg.model.reduction_factor = 4
    elif cfg.model.fps == 50:
        cfg.model.reduction_factor = 2

    if cfg.model.imsize_cropped == 96:
        cfg.model.is_large = True
    elif cfg.model.imsize_cropped == 48:
        cfg.model.is_large = False


def get_path_train(cfg, current_time):
    if cfg.train.face_or_lip == 'avhubert_preprocess_fps25_gray':
        train_data_root = cfg.train.avhubert_preprocess_fps25_train
        val_data_root = cfg.train.avhubert_preprocess_fps25_val

    train_data_root = Path(train_data_root).expanduser()
    val_data_root = Path(val_data_root).expanduser()

    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name

    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / cfg.model.name / ckpt_time
    else:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / cfg.model.name / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.face_or_lip / cfg.model.name / ckpt_time    
    else:
        save_path = save_path / cfg.train.face_or_lip / cfg.model.name / current_time
    os.makedirs(save_path, exist_ok=True)

    return train_data_root, val_data_root, ckpt_path, save_path, ckpt_time


def get_save_and_ckpt_path(
    cfg,
    current_time,
):
    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name
        
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / cfg.model.name / ckpt_time
    else:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / cfg.model.name / current_time
    ckpt_path.mkdir(parents=True, exist_ok=True)

    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.face_or_lip / cfg.model.name / ckpt_time    
    else:
        save_path = save_path / cfg.train.face_or_lip / cfg.model.name / current_time
    save_path.mkdir(parents=True, exist_ok=True)
    return ckpt_path, save_path, ckpt_time


def get_path_train_raw(cfg, current_time):
    if cfg.train.face_or_lip == 'avhubert_preprocess_fps25_gray':
        video_dir = cfg.train.kablab.avhubert_preprocess_fps25_video_dir
    video_dir = Path(video_dir).expanduser()
    audio_dir = Path(cfg.train.kablab.audio_dir).expanduser()

    ckpt_path, save_path, ckpt_time= get_save_and_ckpt_path(cfg, current_time)

    return video_dir, audio_dir, ckpt_path, save_path, ckpt_time


def get_path_test(cfg, model_path):
    if cfg.train.face_or_lip == 'avhubert_preprocess_fps25_gray':
        train_data_root = cfg.train.avhubert_preprocess_fps25_train
        test_data_root = cfg.test.avhubert_preprocess_fps25_test
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / cfg.model.name / model_path.parents[0].name / model_path.stem

    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    train_save_path.mkdir(parents=True, exist_ok=True)
    test_save_path.mkdir(parents=True, exist_ok=True)

    data_root_list = [test_data_root]
    save_path_list = [test_save_path]
    # data_root_list = [train_data_root]
    # save_path_list = [train_save_path]

    return data_root_list, save_path_list, train_data_root


def get_path_test_raw(cfg, model_path):
    if cfg.train.face_or_lip == 'avhubert_preprocess_fps25_gray':
        video_dir = cfg.train.kablab.avhubert_preprocess_fps25_video_dir
    video_dir = Path(video_dir).expanduser()
    audio_dir = Path(cfg.train.kablab.audio_dir).expanduser()

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / cfg.model.name / model_path.parents[0].name / model_path.stem

    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    train_save_path.mkdir(parents=True, exist_ok=True)
    test_save_path.mkdir(parents=True, exist_ok=True)

    return video_dir, audio_dir, test_save_path


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


def get_datasets_raw(cfg, video_dir, audio_dir, data_split):
    data_path_list = []
    if cfg.train.kablab.use:
        print('load kablab')
        df = pd.read_csv(str(Path(cfg.train.kablab.df_path).expanduser()))
        df = df.loc[df['speaker'].isin(cfg.train.speaker)]
        df = df.loc[df['corpus'].isin(cfg.train.corpus)]
        df = df.loc[df['data_split'] == data_split]
        for i in range(df.shape[0]):
            row = df.iloc[i]
            audio_path = audio_dir / row['speaker'] / f'{row["filename"]}.wav'
            video_path = video_dir / row['speaker'] / f'{row["filename"]}.mp4'
            if (not audio_path.exists()) or (not video_path.exists()):
                continue
            data_path_list.append(
                {
                    'audio_path': audio_path,
                    'video_path': video_path,
                    'speaker': row['speaker'],
                    'filename': row['filename'],
                }
            )
    return data_path_list


def get_datasets_external_data_raw(cfg, data_split):
    data_path_list = []
    if cfg.train.tcd_timit.use:
        print('load tcd-timit')
        df = pd.read_csv(str(Path(cfg.train.tcd_timit.df_path).expanduser()))
        df = df.loc[df['data_split'] == data_split]
        audio_dir = Path(cfg.train.tcd_timit.audio_dir).expanduser()
        video_dir = Path(cfg.train.tcd_timit.avhubert_preprocess_fps25_video_dir).expanduser()
        for i in range(df.shape[0]):
            row = df.iloc[i]
            data_path_list.append(
                {
                    'audio_path': audio_dir / row['speaker'] / 'straightcam' / f"{row['filename']}.wav",
                    'video_path': video_dir / row['speaker'] / 'straightcam' / f"{row['filename']}.mp4",
                    'speaker': row['speaker'],
                    'filename': row['filename'], 
                }
            )
    if cfg.train.hifi_captain.use:
        print('load hi-fi-captain')
        df = pd.read_csv(str(Path(cfg.train.hifi_captain.df_path).expanduser()))
        df = df.loc[df['data_split'] == data_split]
        audio_dir = Path(cfg.train.hifi_captain.data_dir).expanduser()
        for i in range(df.shape[0]):
            row = df.iloc[i]
            data_path_list.append(
                {
                    'audio_path': audio_dir / row['speaker'] / 'wav' / row['parent_dir'] / f'{row["filename"]}.wav',
                    'video_path': None,
                    'speaker': row['speaker'],
                    'filename': row['filename'],
                }
            )
    if cfg.train.jvs.use:
        print('load jvs')
        df = pd.read_csv(str(Path(cfg.train.jvs.df_path).expanduser()))
        df = df.loc[
            (df['data'] == 'parallel100') | (df['data'] == 'nonpara30')
        ]
        df = df.loc[df['data_split'] == data_split]
        audio_dir = Path(cfg.train.jvs.data_dir).expanduser()
        for i in range(df.shape[0]):
            row = df.iloc[i]
            data_path_list.append(
                {
                    'audio_path': audio_dir / row['speaker'] / row['data'] / 'wav24kHz16bit' / f'{row["filename"]}.wav',
                    'video_path': None,
                    'speaker': row['speaker'],
                    'filename': row['filename'],
                }
            )
    if cfg.train.vctk.use:
        print('load vctk')
        df = pd.read_csv(str(Path(cfg.train.vctk.df_path).expanduser()))
        df = df.loc[df['data_split'] == data_split]
        audio_dir = Path(cfg.train.vctk.data_dir).expanduser()
        for i in range(df.shape[0]):
            row = df.iloc[i]
            data_path_list.append(
                {
                    'audio_path': audio_dir / row['speaker'] / f'{row["filename"]}.wav',
                    'video_path': None,
                    'speaker': row['speaker'],
                    'filename': row['speaker'],
                }
            )
    return data_path_list


def get_datasets_test(data_root, cfg):
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.test.speaker:
        print(f"load {speaker}")
        spk_path = data_root / speaker / cfg.model.name
        spk_path = list(spk_path.glob("*.npz"))
        items += spk_path
    return items


def get_datasets_test_raw(cfg, video_dir, audio_dir):
    df = pd.read_csv(str(Path(cfg.train.kablab.df_path).expanduser()))
    df = df.loc[df['data_split'] == 'test']
    df = df.loc[df['speaker'].isin(cfg.test.speaker)]
    data_path_list = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        data_path_list.append(
            {
                'audio_path': audio_dir / row['speaker'] / f'{row["filename"]}.wav',
                'video_path': video_dir / row['speaker'] / f'{row["filename"]}.mp4',
                'speaker': row['speaker'],
                'filename': row['filename'],
            }
        )
    return data_path_list


def get_datasets_external_data(cfg):
    items = []
    if cfg.train.which_external_data == "lrs2_main":
        print(f"\n--- get datasets lrs2_main ---")
        data_dir = Path(cfg.train.lrs2_npz_path).expanduser()
        items += list(data_dir.glob(f"*/{cfg.model.name}/*.npz"))
    if cfg.train.which_external_data == "lrs2_pretrain":
        print(f"\n--- get datasets lrs2_pretrain ---")
        data_dir = Path(cfg.train.lrs2_pretrain_npz_path).expanduser()
        items += list(data_dir.glob(f"*/{cfg.model.name}/*.npz"))
    if cfg.train.which_external_data == "lip2wav":
        print(f"\n--- get datasets lip2wav ---")
        data_dir = Path(cfg.train.lip2wav_npz_path).expanduser()
        items += list(data_dir.glob(f"*/{cfg.model.name}/*.npz"))
    if cfg.train.use_jsut_corpus:
        print(f"\n--- get datasets jsut ---")
        data_dir = Path(cfg.train.jsut_path_train).expanduser()
        items += list(data_dir.glob(f"*/{cfg.model.name}/*.npz"))
    if cfg.train.jvs.use:
        print(f"\n--- get datasets jvs ---")
        data_dir = Path(cfg.train.jvs_path_train).expanduser()
        items += list(data_dir.glob(f"*/{cfg.model.name}/*.npz"))
    return items


def make_train_val_loader(cfg, train_data_root, val_data_root):
    # パスを取得
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    if cfg.train.debug:
        train_data_path = train_data_path[:100]
        val_data_path = val_data_path[:100]

    # 学習用，検証用それぞれに対してtransformを作成
    train_trans = KablabTransform(cfg, "train")
    val_trans = KablabTransform(cfg, "val")

    # dataset作成
    print("\n--- make train dataset ---")
    if cfg.train.use_synth_corpus:
        train_data_path_synth = train_data_path
        train_dataset = KablabDataset(
            data_path=train_data_path_synth,
            train_data_path=train_data_path,
            transform=train_trans,
            cfg=cfg,
        )
    else:
        train_dataset = KablabDataset(
            data_path=train_data_path,
            train_data_path=train_data_path,
            transform=train_trans,
            cfg=cfg,
        )
    print("\n--- make validation dataset ---")
    val_dataset = KablabDataset(
        data_path=val_data_path,
        train_data_path=train_data_path,
        transform=val_trans,
        cfg=cfg,
    )

    # それぞれのdata loaderを作成
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def data_upsampling(data_path, upsamling_factor):
    data_path = copy.deepcopy(data_path)
    data_path *= upsamling_factor
    return data_path


def make_train_val_loader_with_external_data(cfg, train_data_root, val_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)
    external_data_path = get_datasets_external_data(cfg)
    train_trans = TransformWithExternalData(cfg, "train")
    val_trans = TransformWithExternalData(cfg, "val")

    if cfg.train.debug:
        train_data_path = train_data_path[:100]
        val_data_path = val_data_path[:100]
        external_data_path = external_data_path[:100]

    if cfg.train.apply_upsampling:
        upsampling_factor = len(external_data_path) // len(train_data_path)
        train_dataset = DatasetWithExternalData(
            data_path=data_upsampling(train_data_path, upsampling_factor) + external_data_path,
            train_data_path=train_data_path,
            transform=train_trans,
            cfg=cfg,
        )
        val_dataset = DatasetWithExternalData(
            data_path=val_data_path,
            train_data_path=train_data_path,
            transform=val_trans,
            cfg=cfg,
        )
    else:
        if cfg.model.avhubert_audio_pretrain:
            # jsutで事前学習したときに統計量を揃えるための分岐
            if len(val_data_path) == 0:
                # jsutだけで事前学習するときは検証データもjsutを使う
                val_data_dir = Path(cfg.train.jsut_path_val).expanduser()
                val_data_path = list(val_data_dir.glob(f"*/{cfg.model.name}/*.npz"))
                train_dataset = DatasetWithExternalData(
                    data_path=external_data_path,
                    train_data_path=external_data_path,
                    transform=train_trans,
                    cfg=cfg,
                )
                val_dataset = DatasetWithExternalData(
                    data_path=val_data_path,
                    train_data_path=external_data_path,
                    transform=val_trans,
                    cfg=cfg,
                )
            else:
                train_dataset = DatasetWithExternalData(
                    data_path=train_data_path,
                    train_data_path=external_data_path,
                    transform=train_trans,
                    cfg=cfg,
                )
                val_dataset = DatasetWithExternalData(
                    data_path=val_data_path,
                    train_data_path=external_data_path,
                    transform=val_trans,
                    cfg=cfg,
                )
        else:
            train_dataset = DatasetWithExternalData(
                data_path=train_data_path + external_data_path,
                train_data_path=train_data_path,
                transform=train_trans,
                cfg=cfg,
            )
            val_dataset = DatasetWithExternalData(
                data_path=val_data_path,
                train_data_path=train_data_path,
                transform=val_trans,
                cfg=cfg,
            )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_with_external_data, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_with_external_data, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_with_external_data_raw(cfg, video_dir, audio_dir):
    train_data_path_list = get_datasets_raw(cfg, video_dir, audio_dir, 'train')
    val_data_path_list = get_datasets_raw(cfg, video_dir, audio_dir, 'val')
    train_external_data_path_list = get_datasets_external_data_raw(cfg, 'train')
    val_external_data_path_list = get_datasets_external_data_raw(cfg, 'val')
    train_trans = TransformWithExternalDataRaw(cfg, "train")
    val_trans = TransformWithExternalDataRaw(cfg, "val")

    if cfg.train.debug:
        train_data_path_list = train_data_path_list[:100]
        val_data_path_list = val_data_path_list[:100]
        train_external_data_path_list = train_external_data_path_list[:100]
        val_external_data_path_list = val_external_data_path_list[:100]

    if cfg.train.tcd_timit.use or cfg.train.vctk.use or cfg.train.jvs.use or cfg.train.hifi_captain.use:
        train_dataset = DatasetWithExternalDataRaw(
            data_path=train_external_data_path_list,
            transform=train_trans,
            cfg=cfg,
        )
        val_dataset = DatasetWithExternalDataRaw(
            data_path=val_external_data_path_list,
            transform=val_trans,
            cfg=cfg,
        )
    else:
        train_dataset = DatasetWithExternalDataRaw(
            data_path=train_data_path_list,
            transform=train_trans,
            cfg=cfg,
        )
        val_dataset = DatasetWithExternalDataRaw(
            data_path=val_data_path_list,
            transform=val_trans,
            cfg=cfg,
        )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_with_external_data, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_with_external_data, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_text(cfg):
    df = []
    if cfg.train.wiki.use:
        df_wiki = pd.read_csv(str(Path(cfg.train.wiki.df_path).expanduser()))
        df.append(df_wiki)
    if cfg.train.aozora.use:
        df_aozora = pd.read_csv(str(Path(cfg.train.aozora.df_path).expanduser()))
        df.append(df_aozora)
    df = pd.concat(df, axis=0, ignore_index=True)
    train_data_path_list = df.loc[df['data_split'] == 'train', 'data_path'].values
    val_data_path_list = df.loc[df['data_split'] == 'val', 'data_path'].values
    
    train_trans = TransformBART(cfg, 'train')
    val_trans = TransformBART(cfg, 'val')
    train_dataset = DatasetBART(train_data_path_list, cfg, train_trans)
    val_dataset = DatasetBART(val_data_path_list, cfg, val_trans)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_bart, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_bart, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_test_loader(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

    if cfg.test.debug:
        train_data_path = train_data_path[:100]

        test_data_path_for_debug = []
        n_data_per_speaker =  defaultdict(int)
        for data_path in test_data_path:
            speaker = data_path.parents[1].name
            if n_data_per_speaker[speaker] >= 1:
                continue
            test_data_path_for_debug.append(data_path)
            n_data_per_speaker[speaker] += 1

        test_data_path = test_data_path_for_debug

    test_trans = KablabTransform(cfg, "test")
    test_dataset = KablabDataset(
        data_path=test_data_path,
        train_data_path=train_data_path,
        transform=test_trans,
        cfg=cfg,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, test_dataset


def make_test_loader_with_external_data(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)
    external_data_path = get_datasets_external_data(cfg)

    if cfg.test.debug:
        train_data_path = train_data_path[:100]
        test_data_path_for_debug = []
        n_data_per_speaker =  defaultdict(int)
        for data_path in test_data_path:
            speaker = data_path.parents[1].name
            if n_data_per_speaker[speaker] >= 1:
                continue
            test_data_path_for_debug.append(data_path)
            n_data_per_speaker[speaker] += 1
        test_data_path = test_data_path_for_debug
        external_data_path = external_data_path[:100]

    test_trans = TransformWithExternalData(cfg, 'test')

    if cfg.model.avhubert_audio_pretrain:
        test_dataset = DatasetWithExternalData(
            data_path=test_data_path,
            train_data_path=external_data_path,
            transform=test_trans,
            cfg=cfg,
        )
    else:
        test_dataset = DatasetWithExternalData(
            data_path=test_data_path,
            train_data_path=train_data_path,
            transform=test_trans,
            cfg=cfg,
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, test_dataset


def make_test_loader_with_external_data_raw(cfg, video_dir, audio_dir):
    train_data_path_list = get_datasets_raw(cfg, video_dir, audio_dir, 'train')
    test_data_path_list = get_datasets_test_raw(cfg, video_dir, audio_dir)
    train_external_data_path_list = get_datasets_external_data_raw(cfg, 'train')
    test_external_data_path_list = get_datasets_external_data_raw(cfg, 'test')

    if cfg.train.tcd_timit.use:
        test_data_path_list = test_external_data_path_list
    
    if cfg.test.debug:
        train_data_path_list = train_data_path_list[:100]
        train_external_data_path_list = train_external_data_path_list[:100]

        test_data_path_list_debug = []
        n_data_per_speaker = defaultdict(int)
        for data_path in test_data_path_list:
            speaker = data_path['speaker']
            if n_data_per_speaker[speaker] > 0:
                continue
            test_data_path_list_debug.append(data_path)
            n_data_per_speaker[speaker] += 1
        test_data_path_list = test_data_path_list_debug
    
    test_trans = TransformWithExternalDataRaw(cfg, 'test')
    test_dataset = DatasetWithExternalDataRaw(
        data_path=test_data_path_list,
        transform=test_trans,
        cfg=cfg,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, test_dataset


def count_params(module, attr):
    """
    モデルパラメータを計算
    """
    params = 0
    for p in module.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"{attr}_parameter = {params}")


def requires_grad_change(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad
    return net


def set_requires_grad_by_name(model, condition, requires_grad):
    for name, param in model.named_parameters():
        if condition(name):
            param.requires_grad = requires_grad


def save_loss(train_loss_list, val_loss_list, save_path, filename):
    loss_save_path = save_path / f"{filename}.png"
    
    plt.figure()
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.plot(np.arange(len(train_loss_list)), val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train loss", "validation loss"])
    plt.grid()
    plt.savefig(str(loss_save_path))
    plt.close("all")

    wandb.log({f"loss {filename}": wandb.plot.line_series(
        xs=np.arange(len(train_loss_list)), 
        ys=[train_loss_list, val_loss_list],
        keys=["train loss", "validation loss"],
        title=f"{filename}",
        xname="epoch",
    )})


def check_mel_ss(orig, mixed, cfg, filename, current_time, ckpt_time=None):
    orig = orig.to("cpu").detach().numpy().copy()
    mixed = mixed.to("cpu").detach().numpy().copy()

    plt.figure(figsize=(7.5, 8))
    plt.subplot(2, 1, 1)
    specshow(
        data=orig, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("original")

    plt.subplot(2, 1, 2)
    specshow(
        data=mixed, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("mixed")

    plt.tight_layout()

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def check_mel_ar(target, output, dec_output, cfg, filename, current_time, ckpt_time=None):
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()
    dec_output = dec_output.to('cpu').detach().numpy().copy()

    plt.figure(figsize=(7.5, 7.5*1.6), dpi=200)
    ax = plt.subplot(3, 1, 1)
    specshow(
        data=target, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("target")
    
    ax = plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=output, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("output")

    ax = plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
    specshow(
        data=dec_output, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("dec_output")

    plt.tight_layout()
    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def check_mel_nar(target, output, cfg, filename, current_time, ckpt_time=None):
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()
    
    plt.close("all")
    plt.figure()
    ax = plt.subplot(2, 1, 1)
    specshow(
        data=target, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("target")
    
    ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=output, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("output")

    plt.tight_layout()
    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def check_f0(target, output, cfg, filename, current_time, ckpt_time=None):
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()
    time = np.arange(target.shape[-1]) / 100

    plt.close("all")
    plt.figure()
    plt.plot(time, target, label="target")
    plt.plot(time, output, label="output")
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.2)
    plt.xlabel("Time[s]")
    plt.title("f0")
    plt.grid()

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def check_wav(target, output, cfg, filename_mel, filename_wav_target, filename_wav_output, current_time, ckpt_time=None):
    """
    target, output : (1, T)
    """
    target = target.squeeze(0)
    output = output.squeeze(0)
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()
    target /= np.max(np.abs(target))
    output /= np.max(np.abs(output))
    wandb.log({f"{filename_wav_target}": wandb.Audio(target, sample_rate=cfg.model.sampling_rate)})
    wandb.log({f"{filename_wav_output}": wandb.Audio(output, sample_rate=cfg.model.sampling_rate)})

    target = wav2mel(target, cfg, ref_max=True)
    output = wav2mel(output, cfg, ref_max=True)
    
    plt.close("all")
    plt.figure()
    ax = plt.subplot(2, 1, 1)
    specshow(
        data=target, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("target")
    
    ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=output, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("output")

    plt.tight_layout()
    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename_mel}.png"))
    wandb.log({f"{filename_mel}": wandb.Image(str(save_path / f"{filename_mel}.png"))})


def check_movie(target, output, lip_mean, lip_std, cfg, filename, current_time, ckpt_time=None):
    """
    target, output : (C, H, W, T)
    """
    target = target.to('cpu').detach()
    output = output.to('cpu').detach()
    lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)
    lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
    target = torch.add(torch.mul(target, lip_std), lip_mean)
    output = torch.add(torch.mul(output, lip_std), lip_mean)
    target = target.permute(-1, 0, 1, 2).to(torch.uint8)  # (T, C, H, W)
    output = output.permute(-1, 0, 1, 2).to(torch.uint8)  # (T, C, H, W)
    if cfg.model.gray:
        target = target.expand(-1, 3, -1, -1)
        output = output.expand(-1, 3, -1, -1)

    wandb.log({f"{filename}_target": wandb.Video(target.numpy(), fps=cfg.model.fps, format="mp4")})
    wandb.log({f"{filename}_output": wandb.Video(output.numpy(), fps=cfg.model.fps, format="mp4")})


def check_text(target, output, epoch, cfg, classes_index, filename, current_time, ckpt_time=None):
    """
    target : (T,)
    output : (C, T)
    """
    target = target.to("cpu").detach().numpy()
    output = output.max(dim=0)[1]   # (T,)
    output = output.to("cpu").detach().numpy()

    phoneme_answer = [get_keys_from_value(classes_index, i) for i in target]
    phoneme_answer = " ".join(phoneme_answer)
    phoneme_predict = [get_keys_from_value(classes_index, i) for i in output]
    phoneme_predict = " ".join(phoneme_predict)

    phoneme_error_rate = wer(phoneme_answer, phoneme_predict)

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    
    with open(str(save_path / f"{filename}.txt"), "a") as f:
        f.write(f"\n--- epoch {epoch} ---\n")
        f.write("answer\n")
        f.write(f"{phoneme_answer}\n")
        f.write("\npredict\n")
        f.write(f"{phoneme_predict}\n")
        f.write(f"\nphoneme error rate = {phoneme_error_rate}\n")


def check_attention_weight(att_w, cfg, filename, current_time, ckpt_time=None):
    """
    att_w : (T, T)
    """
    att_w = att_w.to('cpu').detach().numpy().copy()

    plt.figure()
    sns.heatmap(att_w, cmap="viridis", cbar=True)
    plt.title("attention weight")
    plt.xlabel("encoder output")
    plt.ylabel("feature")

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    plt.close()
    
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def mixing_prob_controller(cfg):
    prob_list = []
    mixing_prob = 0
    for i in range(cfg.train.max_epoch):
        mixing_prob = cfg.train.exp_factor ** i
        if mixing_prob > cfg.train.min_mixing_prob:
            prob_list.append(mixing_prob)
        else:
            prob_list.append(cfg.train.min_mixing_prob)
    return prob_list


def mixing_prob_controller_f0(cfg):
    prob_list = []
    mixing_prob = 0
    for i in range(cfg.train.max_epoch):
        mixing_prob = cfg.train.exp_factor_f0 ** i
        if mixing_prob > cfg.train.min_mixing_prob_f0:
            prob_list.append(mixing_prob)
        else:
            prob_list.append(cfg.train.min_mixing_prob_f0)
    return prob_list


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def select_checkpoint(cfg):
    '''
    checkpointの中から最も検証データに対しての損失が小さいものを選ぶ
    '''
    checkpoint_path_last = Path(cfg.test.model_path).expanduser()
    checkpoint_dict_last = torch.load(str(checkpoint_path_last))
    best_checkpoint = np.argmin(checkpoint_dict_last[cfg.test.metric_for_select]) + 1
    filename_prev = checkpoint_path_last.stem + checkpoint_path_last.suffix
    filename_new = str(best_checkpoint) + checkpoint_path_last.suffix
    checkpoint_path = Path(str(checkpoint_path_last).replace(filename_prev, filename_new))
    return checkpoint_path


def load_pretrained_model(model_path, model, model_name):
    model_dict = model.state_dict()

    if model_path.suffix == ".ckpt":
        if torch.cuda.is_available():
            pretrained_dict = torch.load(str(model_path))[str(model_name)]
        else:
            pretrained_dict = torch.load(str(model_path), map_location=torch.device('cpu'))[str(model_name)]
    elif model_path.suffix == ".pth":
        if torch.cuda.is_available():
            pretrained_dict = torch.load(str(model_path))
        else:
            pretrained_dict = torch.load(str(model_path), map_location=torch.device('cpu'))

    pretrained_dict = fix_model_state_dict(pretrained_dict)
    match_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(match_dict)
    model.load_state_dict(match_dict)
    return model


def gen_data_separate(data, input_length, shift_frame):
    """
    data : (..., T)
    """
    start_frame = 0
    data_list = []
    while True:
        if data.shape[-1] <= start_frame + input_length:
            data_list.append(data[..., -input_length:])
            break
        else:
            data_list.append(data[..., start_frame:start_frame + input_length])

        start_frame += shift_frame

    data = torch.cat(data_list, dim=0)
    return data


def gen_data_concat(data, shift_frame, n_last_frame):
    """
    data : (B, ..., T)
    """
    data_list = []

    for i in range(data.shape[0]):
        if i == 0:
            data_list.append(data[i, ...])

        elif i == data.shape[0] - 1:
            if n_last_frame != 0:
                data_list.append(data[i, ..., -n_last_frame:])
            else:
                data_list.append(data[i, ..., -shift_frame:])

        else:
            data_list.append(data[i, ..., -shift_frame:])

    data = torch.cat(data_list, dim=-1).unsqueeze(0)     # (1, C, T)
    return data


def delete_unnecessary_checkpoint(result_dir, checkpoint_dir):
    checkpoint_dir_list = list(checkpoint_dir.glob('*'))
    for ckpt_dir in checkpoint_dir_list:
        ckpt_path_list = list(ckpt_dir.glob('*'))
        result_path = result_dir / ckpt_dir.stem
        if not result_path.exists():
            continue
        result_path = list(result_path.glob('*'))[0]
        required_ckpt_filename = result_path.stem
        for ckpt_path in ckpt_path_list:
            if ckpt_path.stem == required_ckpt_filename:
                continue
            os.remove(str(ckpt_path))


def calc_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def load_raven(cfg):
    '''
    cfg: cfg.raven_config
    '''
    if cfg.model_size == 'base':
        raven = MyRAVEN(1, cfg.base).encoder
        ckpt_path = Path(cfg.base.ckpt_path).expanduser()
    elif cfg.model_size == 'large':
        raven = MyRAVEN(1, cfg.large).encoder
        ckpt_path = Path(cfg.large.ckpt_path).expanduser()

    if cfg.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))
        model_dict = raven.state_dict()
        match_dict = {name: params for name, params in pretrained_dict.items() if name in model_dict}
        raven.load_state_dict(match_dict, strict=True)

    return raven


def load_vatlm(cfg):
    '''
    cfg: cfg.vatlm_config
    '''
    if cfg.model_size == 'base':
        vatlm = MyVATLM(
            cfg=cfg.base.cfg,
            task_cfg=cfg.base.task_cfg,
            dictionaries=None,
        )
        ckpt_path = Path(cfg.base.ckpt_path).expanduser()
    elif cfg.model_size == 'large':
        vatlm = MyVATLM(
            cfg=cfg.large.cfg,
            task_cfg=cfg.large.task_cfg,
            dictionaries=None,
        )
        ckpt_path = Path(cfg.large.ckpt_path).expanduser()

    if cfg.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))['vatlm']
        vatlm.load_state_dict(pretrained_dict, strict=True)

    return vatlm


def load_avhubert(cfg):
    '''
    cfg: cfg.avhubert_config
    '''
    if cfg.model_size == 'base':
        avhubert = MyAVHubertModel(cfg.base)
        ckpt_path = Path(cfg.base.ckpt_path).expanduser()
    elif cfg.model_size == 'large':
        avhubert = MyAVHubertModel(cfg.large)
        ckpt_path = Path(cfg.large.ckpt_path).expanduser()
    
    if cfg.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))['avhubert']
        avhubert.load_state_dict(pretrained_dict, strict=True)
    
    return avhubert


def load_ssl_ensemble(
        model,
        ckpt_path,
        device,
        ssl_model_name,
):
    '''
    ssl_model_name: 'avhubert.', 'raven.', etc ...
    '''
    ckpt = torch.load(str(ckpt_path), map_location=device)['model']
    ckpt = {name: param for name, param in ckpt.items() if ssl_model_name in name}
    model_dict = model.state_dict()
    match_dict = {name: param for name, param in ckpt.items() if name in model_dict}
    model.load_state_dict(match_dict, strict=False)
    return model