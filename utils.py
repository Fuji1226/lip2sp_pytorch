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

from dataset.dataset_npz import KablabDataset, KablabTransform, collate_time_adjust


def get_upsample(fps, fs, frame_period):
    """
    動画のfpsと音響特徴量のフレームあたりの秒数から対応関係を求める
    """
    nframes = 1000 // frame_period
    upsample = nframes // fps
    return int(upsample)


def set_config(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 1
        cfg.train.num_workers = 1

    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False

    # if cfg.train.use_time_augment or cfg.train.use_frame_masking or cfg.train.use_segment_masking:
    #     cfg.model.delta = False
    #     cfg.model.in_channels = 3


def get_path_train(cfg, current_time):
    # data
    if cfg.train.face_or_lip == "lip":
        print("use lip")
        train_data_root = cfg.train.lip_pre_loaded_path_train
        val_data_root = cfg.train.lip_pre_loaded_path_val
        stat_path = cfg.train.lip_stat_path
    if cfg.train.face_or_lip == "lip_st":
        print("use lip_st")
        train_data_root = cfg.train.lip_pre_loaded_path_train_st
        val_data_root = cfg.train.lip_pre_loaded_path_val_st
        stat_path = cfg.train.lip_stat_path_st

    train_data_root = Path(train_data_root).expanduser()
    val_data_root = Path(val_data_root).expanduser()
    stat_path = Path(stat_path).expanduser()

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

    return train_data_root, val_data_root, stat_path, ckpt_path, save_path, ckpt_time


def get_path_test(cfg, model_path):
    if cfg.test.face_or_lip == "lip":
        train_data_root = cfg.train.lip_pre_loaded_path_train
        test_data_root = cfg.test.lip_pre_loaded_path
        stat_path = cfg.train.lip_stat_path
    if cfg.test.face_or_lip == "lip_st":
        train_data_root = cfg.train.lip_pre_loaded_path_train_st
        test_data_root = cfg.test.lip_pre_loaded_path_st
        stat_path = cfg.train.lip_stat_path_st
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()
    stat_path = Path(stat_path).expanduser()

    save_path = Path(cfg.test.save_path).expanduser()
    
    if cfg.model.name == "mspec":
        if cfg.model.sharp:
            save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / f"{model_path.stem}_sharp"
        else:
            save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem
    else:
        save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem

    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    # data_root_list = [test_data_root, train_data_root]
    # save_path_list = [test_save_path, train_save_path]

    data_root_list = [test_data_root]
    save_path_list = [test_save_path]

    return data_root_list, stat_path, save_path_list


def get_path_test_vc(cfg, model_path, speaker, reference):
    if cfg.test.face_or_lip == "lip":
        train_data_root = cfg.train.lip_pre_loaded_path_train
        test_data_root = cfg.test.lip_pre_loaded_path
        stat_path = cfg.train.lip_stat_path
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()
    stat_path = Path(stat_path).expanduser()

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem
    train_save_path = save_path / "train_data" / str(speaker) / str(reference) / "audio"
    test_save_path = save_path / "test_data" / str(speaker) / str(reference) / "audio"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    # data_root_list = [test_data_root, train_data_root]
    # save_path_list = [test_save_path, train_save_path]

    data_root_list = [test_data_root]
    save_path_list = [test_save_path]
    return data_root_list, stat_path, save_path_list


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


def get_datasets(data_root, cfg):    
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.train.speaker:
        print(f"{speaker}")
        spk_path_list = []
        spk_path = data_root / speaker

        for corpus in cfg.train.corpus:
            spk_path_co = [p for p in spk_path.glob(f"*{cfg.model.name}.npz") if re.search(f"{corpus}", str(p))]
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
        spk_path = data_root / speaker
        spk_path = list(spk_path.glob(f"*{cfg.model.name}.npz"))
        items += spk_path
    return items


def make_train_val_loader(cfg, train_data_root, val_data_root, stat_path):
    # パスを取得
    train_data_path = get_datasets(train_data_root, cfg)
    val_data_path = get_datasets(val_data_root, cfg)

    # 学習用，検証用それぞれに対してtransformを作成
    train_trans = KablabTransform(
        cfg=cfg,
        train_val_test="train",
    )
    val_trans = KablabTransform(
        cfg=cfg,
        train_val_test="val",
    )

    # dataset作成
    print("\n--- make train dataset ---")
    train_dataset = KablabDataset(
        data_path=train_data_path,
        stat_path = stat_path,
        transform=train_trans,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = KablabDataset(
        data_path=val_data_path,
        stat_path=stat_path,
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


def make_test_loader(cfg, data_root, stat_path):
    test_data_path = get_datasets_test(
        data_root=data_root,
        cfg=cfg,
    )
    test_data_path = sorted(test_data_path)
    test_trans = KablabTransform(
        cfg=cfg,
        train_val_test="test",
    )
    test_dataset = KablabDataset(
        data_path=test_data_path,
        stat_path = stat_path,
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


def calc_class_balance(cfg, data_root, device):
    """
    話者ごとのデータ量の偏りを計算
    """
    # data_path = get_datasets(
    #     data_root=data_root,
    #     cfg=cfg,
    # )
    print("\ncalc_class_balance")
    data_path = {}
    for speaker in cfg.train.speaker:
        print(f"{speaker}")
        spk_path_list = []
        spk_path = data_root / speaker

        for corpus in cfg.train.corpus:
            spk_path_co = [p for p in spk_path.glob(f"*{cfg.model.name}.npz") if re.search(f"{corpus}", str(p))]
            if len(spk_path_co) > 1:
                print(f"load {corpus}")
            spk_path_list += spk_path_co

        data_path[speaker] = spk_path_list

    num_data = []
    for key, value in data_path.items():
        print(f"{key} : {len(value)}")
        num_data.append(len(value))    
    
    num_data_max = max(num_data)
    class_weight = torch.tensor([num_data_max / d for d in num_data])
    print(f"class_weight = {class_weight}\n")
    return class_weight.to(device)


def count_params(module, attr):
    """
    モデルパラメータを計算
    """
    params = 0
    for p in module.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"{attr}_parameter = {params}")


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
    

def check_feat_add(target, output, cfg, filename, current_time, ckpt_time=None):
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()
    f0_target = target[0]
    f0_output = output[0]
    # power_target = target[1]
    # power_output = output[1]
    time = np.arange(target.shape[-1]) / 100

    plt.close("all")
    plt.figure()
    plt.plot(time, f0_target, label="target")
    plt.plot(time, f0_output, label="output")
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.2)
    plt.xlabel("Time[s]")
    plt.title("f0")
    plt.grid()

    # ax = plt.subplot(2, 1, 2)
    # ax.plot(time, power_target, label="target")
    # ax.plot(time, power_output, label="output")
    # plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.2)
    # plt.xlabel("Time[s]")
    # plt.title("power")
    # plt.grid()

    # plt.tight_layout()
    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def check_attention_weight(att_w, cfg, filename, current_time, ckpt_time=None):
    att_w = att_w.to('cpu').detach().numpy().copy()
    plt.close()
    plt.matshow(att_w, cmap="viridis")
    plt.colorbar()
    plt.title("attention weight")

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def gen_separate(lip, input_length, shift_frame):
    """
    合成時に系列長を学習時と揃えるための処理
    lip : (B, C, H, W, T)
    input_length : モデル学習時の系列長
    shift_frame : シフト幅
    """
    _, C, H, W, _ = lip.shape
    start_frame = 0
    lip_list = []

    while True:
        if lip.shape[-1] <= start_frame + input_length:
            lip_list.append(lip[..., -input_length:])
            break
        else:
            lip_list.append(lip[..., start_frame:start_frame + input_length])
        
        start_frame += shift_frame

    lip = torch.cat(lip_list, dim=0)    # (B, C, H, W, T)
    return lip


def gen_cat_feature(feature, shift_frame, n_last_frame, upsample):
    """
    gen_separateで分割して出力した結果を結合
    feature : (B, C, T)
    input_length : モデル学習時の系列長
    shift_frame : シフト幅
    """
    feat_list = []
    shift_frame = int(shift_frame * upsample)
    n_last_frame = int(n_last_frame * upsample)

    for i in range(feature.shape[0]):
        if i == 0:
            feat_list.append(feature[i, ...])

        elif i == feature.shape[0] - 1:
            if n_last_frame != 0:
                feat_list.append(feature[i, :, -n_last_frame:])
            else:
                feat_list.append(feature[i, :, -shift_frame:])

        else:
            feat_list.append(feature[i, :, -shift_frame:])

    feature = torch.cat(feat_list, dim=-1).unsqueeze(0)     # (1, C, T)
    return feature
    
