import re
import os
from pathlib import Path
import numpy as np
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
from collections import OrderedDict
import pandas as pd

from dataset.dataset_npz import KablabDataset, KablabTransform, collate_time_adjust, collate_time_adjust_tts
from dataset.dataset_npz_ssl import KablabDatasetSSL, KablabTransformSSL, collate_time_adjust_ssl
from dataset.dataset_lipreading import LipReadingDataset, LipReadingTransform, collate_time_adjust_lipreading
from dataset.dataset_tts import DatasetTTS, TransformTTS
# from dataset.dataset_tts_face import DatasetTTSFace, TransformTTSFace, collate_time_adjust_tts_face
from dataset.dataset_lm import DatasetLM, TransformLM, collate_time_adjust_lm
from dataset.dataset_lrs2 import LRS2Dataset, LRS2Transform, collate_time_adjust_lrs2
from dataset.dataset import Lip2spDataset, Lip2spTransform, collate_time_adjust_lip2sp
from dataset.dataset_tts_face_raw import DatasetTTSFace, TransformTTSFace, collate_time_adjust_tts_face
from data_process.feature import wav2mel
from data_process.phoneme_encode import get_keys_from_value


def get_padding(kernel_size, dilation=1):
    return (kernel_size*dilation - dilation) // 2


def set_config(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
        cfg.train.num_workers = 1
        cfg.train.corpus = ["ATR"]

    if cfg.model.fps == 25:
        cfg.model.reduction_factor = 4
    elif cfg.model.fps == 50:
        cfg.model.reduction_factor = 2

    if cfg.model.imsize_cropped == 96:
        cfg.model.is_large = True
    elif cfg.model.imsize_cropped == 48:
        cfg.model.is_large = False


def get_path_train(cfg, current_time):
    # data
    if cfg.train.face_or_lip == "lip_cropped_0.3_50_gray":
        train_data_root = cfg.train.lip_pre_loaded_path_train_03_50_gray
        val_data_root = cfg.train.lip_pre_loaded_path_val_03_50_gray
    elif cfg.train.face_or_lip == "lip_cropped_0.8_50_gray":
        train_data_root = cfg.train.lip_pre_loaded_path_train_08_50_gray
        val_data_root = cfg.train.lip_pre_loaded_path_val_08_50_gray
    elif cfg.train.face_or_lip == "face_aligned_0_50_gray":
        train_data_root = cfg.train.face_pre_loaded_path_train_0_50_gray
        val_data_root = cfg.train.face_pre_loaded_path_val_0_50_gray
    elif cfg.train.face_or_lip == "face_aligned_0_50":
        train_data_root = cfg.train.face_pre_loaded_path_train_0_50
        val_data_root = cfg.train.face_pre_loaded_path_val_0_50

    train_data_root = Path(train_data_root).expanduser()
    val_data_root = Path(val_data_root).expanduser()

    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name

    # check point
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / ckpt_time
    else:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    # save
    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.face_or_lip / ckpt_time    
    else:
        save_path = save_path / cfg.train.face_or_lip / current_time
    os.makedirs(save_path, exist_ok=True)

    return train_data_root, val_data_root, ckpt_path, save_path, ckpt_time


def get_path_train_raw(cfg, current_time):
    data_dir = Path(cfg.train.data_dir).expanduser()
    bbox_dir = Path(cfg.train.bbox_dir).expanduser()
    landmark_dir = Path(cfg.train.landmark_dir).expanduser()
    train_df_path = Path(cfg.train.train_df_path).expanduser()
    val_df_path = Path(cfg.train.val_df_path).expanduser()
    train_df = pd.read_csv(str(train_df_path), header=None)
    val_df = pd.read_csv(str(val_df_path), header=None)

    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name

    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / ckpt_time
    else:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.face_or_lip / ckpt_time    
    else:
        save_path = save_path / cfg.train.face_or_lip / current_time
    os.makedirs(save_path, exist_ok=True)
    return data_dir, bbox_dir, landmark_dir, train_df, val_df, ckpt_path, save_path, ckpt_time


def get_path_train_lrs2(cfg, current_time):
    train_data_root = Path(cfg.train.lrs2_train_data_root).expanduser()
    val_data_root = Path(cfg.train.lrs2_val_data_root).expanduser()
    train_data_bbox_root = Path(cfg.train.lrs2_train_data_bbox_root).expanduser()
    val_data_bbox_root = Path(cfg.train.lrs2_val_data_bbox_root).expanduser()
    train_data_landmark_root = Path(cfg.train.lrs2_train_data_landmark_root).expanduser()
    val_data_landmark_root = Path(cfg.train.lrs2_val_data_landmark_root).expanduser()
    train_df_path = Path(cfg.train.lrs2_train_df_path).expanduser()
    val_df_path = Path(cfg.train.lrs2_val_df_path).expanduser()

    train_data_df = pd.read_csv(str(train_df_path), header=None)
    val_data_df = pd.read_csv(str(val_df_path), header=None)

    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name

    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / ckpt_time
    else:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.face_or_lip / ckpt_time    
    else:
        save_path = save_path / cfg.train.face_or_lip / current_time
    os.makedirs(save_path, exist_ok=True)
    return train_data_root, train_data_bbox_root, train_data_landmark_root, train_data_df, \
        val_data_root, val_data_bbox_root, val_data_landmark_root, val_data_df, ckpt_path, save_path, ckpt_time


def get_path_test(cfg, model_path):
    if cfg.train.face_or_lip == "lip_cropped_0.3_50_gray":
        train_data_root = cfg.train.lip_pre_loaded_path_train_03_50_gray
        test_data_root = cfg.test.lip_pre_loaded_path_03_50_gray
    elif cfg.train.face_or_lip == "lip_cropped_0.8_50_gray":
        train_data_root = cfg.train.lip_pre_loaded_path_train_08_50_gray
        test_data_root = cfg.test.lip_pre_loaded_path_08_50_gray
    elif cfg.train.face_or_lip == "face_aligned_0_50_gray":
        train_data_root = cfg.train.face_pre_loaded_path_train_0_50_gray
        test_data_root = cfg.test.face_pre_loaded_path_0_50_gray
    elif cfg.train.face_or_lip == "face_aligned_0_50":
        train_data_root = cfg.train.face_pre_loaded_path_train_0_50
        test_data_root = cfg.test.face_pre_loaded_path_0_50
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem

    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    data_root_list = [test_data_root]
    save_path_list = [test_save_path]
    # data_root_list = [train_data_root]
    # save_path_list = [train_save_path]

    return data_root_list, save_path_list, train_data_root


def get_path_test_raw(cfg, model_path):
    data_dir = Path(cfg.train.data_dir).expanduser()
    bbox_dir = Path(cfg.train.bbox_dir).expanduser()
    landmark_dir = Path(cfg.train.landmark_dir).expanduser()
    train_df_path = Path(cfg.train.train_df_path).expanduser()
    test_df_path = Path(cfg.test.test_df_path).expanduser()
    train_df = pd.read_csv(str(train_df_path), header=None)
    test_df = pd.read_csv(str(test_df_path), header=None)

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem

    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    df_list = [test_df]
    save_path_list = [test_save_path]
    return data_dir, bbox_dir, landmark_dir, df_list, save_path_list, train_df


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


def get_datasets_test(data_root, cfg):
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.test.speaker:
        print(f"load {speaker}")
        spk_path = data_root / speaker / cfg.model.name
        spk_path = list(spk_path.glob("*.npz"))
        items += spk_path
    return items


def make_train_val_loader(cfg, train_data_root, val_data_root):
    # パスを取得
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    # 学習用，検証用それぞれに対してtransformを作成
    train_trans = KablabTransform(cfg, "train")
    val_trans = KablabTransform(cfg, "val")

    # dataset作成
    print("\n--- make train dataset ---")
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
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_raw(data_dir, bbox_dir, landmark_dir, train_df, val_df, cfg):
    train_trans = Lip2spTransform(cfg, "train")
    val_trans = Lip2spTransform(cfg, "val")

    print(f"\n--- make train dataset ---")
    train_dataset = Lip2spDataset(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=train_df,
        train_df=train_df,
        transform=train_trans,
        cfg=cfg,
    )
    print(f"\n--- make val dataset ---")
    val_dataset = Lip2spDataset(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=val_df,
        train_df=train_df,
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
        collate_fn=partial(collate_time_adjust_lip2sp, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_lip2sp, cfg=cfg),
    )    
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_lrs2(
    cfg, train_data_root, train_data_bbox_root, train_data_landmark_root, train_data_df, 
    val_data_root, val_data_bbox_root, val_data_landmark_root, val_data_df):
    train_trans = LRS2Transform(cfg, "train")
    val_trans = LRS2Transform(cfg, "val")

    print(f"\n--- make train dataset ---")
    train_dataset = LRS2Dataset(
        data_root=train_data_root,
        data_bbox_root=train_data_bbox_root,
        data_landmark_root=train_data_landmark_root,
        data_df=train_data_df,
        train_data_root=train_data_root,
        train_data_bbox_root=train_data_bbox_root,
        train_data_landmark_root=train_data_landmark_root,
        train_data_df=train_data_df,
        transform=train_trans,
        cfg=cfg,
    )
    print(f"\n--- make val dataset ---")
    val_dataset = LRS2Dataset(
        data_root=val_data_root,
        data_bbox_root=val_data_bbox_root,
        data_landmark_root=val_data_landmark_root,
        data_df=val_data_df,
        train_data_root=train_data_root,
        train_data_bbox_root=train_data_bbox_root,
        train_data_landmark_root=train_data_landmark_root,
        train_data_df=train_data_df,
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
        collate_fn=partial(collate_time_adjust_lrs2, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_lrs2, cfg=cfg),
    )    
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_lipreading(cfg, train_data_root, val_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    train_trans = LipReadingTransform(cfg, "train")
    val_trans = LipReadingTransform(cfg, "val")

    print("\n--- make train dataset ---")
    train_dataset = LipReadingDataset(
        data_path=train_data_path,
        train_data_path=train_data_path,
        transform=train_trans,
        cfg=cfg,
    )
    val_dataset = LipReadingDataset(
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
        collate_fn=partial(collate_time_adjust_lipreading, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_lipreading, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_tts(cfg, train_data_root, val_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    train_trans = KablabTransform(cfg, "train")
    val_trans = KablabTransform(cfg, "val")

    print("\n--- make train dataset ---")
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

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_tts, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_tts, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_tts_face_raw(data_dir, bbox_dir, landmark_dir, train_df, val_df, cfg):
    train_trans = TransformTTSFace(cfg, "train")
    val_trans = TransformTTSFace(cfg, "val")

    print("\n--- make train dataset ---")
    train_dataset = DatasetTTSFace(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=train_df,
        train_df=train_df,
        transform=train_trans,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = DatasetTTSFace(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=val_df,
        train_df=train_df,
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
        collate_fn=partial(collate_time_adjust_tts_face, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_tts_face, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_tts_face_raw_ddp(data_dir, bbox_dir, landmark_dir, train_df, val_df, cfg, rank, n_gpu):
    train_trans = TransformTTSFace(cfg, "train")
    val_trans = TransformTTSFace(cfg, "val")

    print("\n--- make train dataset ---")
    train_dataset = DatasetTTSFace(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=train_df,
        train_df=train_df,
        transform=train_trans,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = DatasetTTSFace(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=val_df,
        train_df=train_df,
        transform=val_trans,
        cfg=cfg,
    )

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=n_gpu, 
        rank=rank, 
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=n_gpu, 
        rank=rank, 
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size // n_gpu,   
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        collate_fn=partial(collate_time_adjust_tts_face, cfg=cfg),
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size // n_gpu,   
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        collate_fn=partial(collate_time_adjust_tts_face, cfg=cfg),
        sampler=val_sampler,
    )
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def make_train_val_loader_tts_face_ddp(rank, n_gpu, cfg, train_data_root, val_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    train_trans = TransformTTSFace(cfg, "train")
    val_trans = TransformTTSFace(cfg, "val")

    print("\n--- make train dataset ---")
    train_dataset = DatasetTTSFace(
        data_path=train_data_path,
        train_data_path=train_data_path,
        transform=train_trans,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = DatasetTTSFace(
        data_path=val_data_path,
        train_data_path=train_data_path,
        transform=val_trans,
        cfg=cfg,
    )

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=n_gpu, 
        rank=rank, 
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=n_gpu, 
        rank=rank, 
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size // n_gpu,   
        # shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        # drop_last=True,
        collate_fn=partial(collate_time_adjust_tts_face, cfg=cfg),
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size // n_gpu,   
        # shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        # drop_last=True,
        collate_fn=partial(collate_time_adjust_tts_face, cfg=cfg),
        sampler=val_sampler,
    )
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def make_train_val_loader_lm(cfg, train_data_root, val_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    train_trans = TransformLM(cfg, "train")
    val_trans = TransformLM(cfg, "val")

    print("\n--- make train dataset ---")
    train_dataset = DatasetLM(
        data_path=train_data_path,
        train_data_path=train_data_path,
        transform=train_trans,
        cfg=cfg,
        load_wiki=True
    )
    print("\n--- make validation dataset ---")
    val_dataset = DatasetLM(
        data_path=val_data_path,
        train_data_path=train_data_path,
        transform=train_trans,
        cfg=cfg,
        load_wiki=False
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_lm, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_lm, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_train_val_loader_ssl(cfg, train_data_root, val_data_root):
    # パスを取得
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    # 学習用，検証用それぞれに対してtransformを作成
    train_trans = KablabTransformSSL(cfg, "train")
    val_trans = KablabTransformSSL(cfg, "val")

    # dataset作成
    print("\n--- make train dataset ---")
    train_dataset = KablabDatasetSSL(
        data_path=train_data_path,
        train_data_path=train_data_path,
        transform=train_trans,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = KablabDatasetSSL(
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
        collate_fn=partial(collate_time_adjust_ssl, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust_ssl, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_test_loader(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

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


def make_test_loader_raw(data_dir, bbox_dir, landmark_dir, train_df, test_df, cfg):
    test_trans = Lip2spTransform(cfg, "test")
    test_dataset = Lip2spDataset(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=test_df,
        train_df=train_df,
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


def make_test_loader_lipreading(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

    test_trans = LipReadingTransform(cfg, "test")
    test_dataset = LipReadingDataset(
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


def make_test_loader_tts(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

    test_trans = TransformTTS(cfg, "test")
    test_dataset = DatasetTTS(
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


def make_test_loader_tts_face_raw(data_dir, bbox_dir, landmark_dir, train_df, test_df, cfg):
    test_trans = TransformTTSFace(cfg, "test")
    test_dataset = DatasetTTSFace(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=test_df,
        train_df=train_df,
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


def make_test_loader_tts_face(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

    test_trans = TransformTTSFace(cfg, "test")
    test_dataset = DatasetTTSFace(
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


def make_test_loader_face_gen(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)
    data_path_for_first_frame = get_datasets_test(train_data_root, cfg)

    test_trans = KablabTransform(cfg, "test")
    dataset_for_first_frame = KablabDataset(
        data_path=data_path_for_first_frame,
        train_data_path=train_data_path,
        transform=test_trans,
        cfg=cfg,
    )
    test_dataset = KablabDataset(
        data_path=test_data_path,
        train_data_path=train_data_path,
        transform=test_trans,
        cfg=cfg,
    )

    loader_for_first_frame = DataLoader(
        dataset=dataset_for_first_frame,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
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
    return test_loader, test_dataset, dataset_for_first_frame, loader_for_first_frame


def make_test_loader_face_gen_raw(data_dir, bbox_dir, landmark_dir, train_df, test_df, cfg):
    test_trans = Lip2spTransform(cfg, "test")
    train_dataset = Lip2spDataset(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=train_df,
        train_df=train_df,
        transform=test_trans,
        cfg=cfg,
    )
    test_dataset = Lip2spDataset(
        data_dir=data_dir,
        bbox_dir=bbox_dir,
        landmark_dir=landmark_dir,
        df=test_df,
        train_df=train_df,
        transform=test_trans,
        cfg=cfg,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,   
        shuffle=True,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
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
    return test_loader, test_dataset, train_loader, train_dataset


def make_test_loader_lm(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

    test_trans = TransformLM(cfg, "test")
    test_dataset = DatasetLM(
        data_path=test_data_path,
        train_data_path=train_data_path,
        transform=test_trans,
        cfg=cfg,
        load_wiki=False
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


def make_test_loader_ssl(cfg, data_root, train_data_root):
    train_data_path = get_datasets(train_data_root, cfg)
    test_data_path = get_datasets_test(data_root, cfg)
    test_data_path = sorted(test_data_path)

    test_trans = KablabTransformSSL(cfg, "test")
    test_dataset = KablabDatasetSSL(
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


def count_params(module, attr):
    """
    モデルパラメータを計算
    """
    params = 0
    for p in module.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"{attr}_parameter = {params}")


def requires_grad_change(net, val):
    for param in net.parameters():
        param.requires_grad = val
    return net


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


def check_mel_default(target, output, dec_output, cfg, filename, current_time, ckpt_time=None):
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
    plt.xlabel("text")
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


def load_pretrained_model(model_path, model, model_name):
    """
    学習したモデルの読み込み
    現在のモデルと事前学習済みのモデルで一致した部分だけを読み込むので,モデルが変わっていても以前のパラメータを読み込むことが可能
    """
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
    model_dict.update(match_dict)
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