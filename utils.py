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
import json

from dataset.dataset_npz import KablabDataset, KablabTransform, collate_time_adjust_for_test, get_datasets, get_datasets_test, collate_time_adjust


def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return tuple(sorted(a))


def get_upsample(fps, fs, frame_period):
    nframes = 1000 // frame_period      # frame_period = 10
    upsample = nframes // fps       # fps = 50
    return int(upsample)    # 2


def get_state_name(feature_type, frame_period, dim):
    return feature_type + "_fp" + str(frame_period) + "_dim" + str(dim)


def get_sp_name(name, feature_type, frame_period, nmels=None):
    ret = name + "_" + feature_type + "_fp" + str(frame_period)

    if feature_type == "mspec":
        ret += "_dim" + str(nmels)

    return ret


def get_path_train(cfg, current_time):
    # data
    if cfg.train.face_or_lip == "face":
        data_root = cfg.train.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    elif cfg.train.face_or_lip == "lip":
        data_root = cfg.train.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    data_root = Path(data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

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

    from omegaconf import DictConfig, OmegaConf
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    save_path_json = os.path.join(save_path, 'config.json')
    config_save = open(save_path_json, mode='w')
    json.dump(wandb_cfg, config_save, indent=4)

    return data_root, mean_std_path, ckpt_path, save_path, ckpt_time


def get_path_test(cfg, model_path):
    if cfg.test.face_or_lip == "face":
        train_data_root = cfg.train.face_pre_loaded_path
        test_data_root = cfg.test.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    if cfg.test.face_or_lip == "lip":
        train_data_root = cfg.train.lip_pre_loaded_path
        test_data_root = cfg.test.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path

        #test_data_root = cfg.train.lip_pre_loaded_path
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / cfg.test.face_or_lip / model_path.parents[0].name / model_path.stem
    train_save_path = save_path / "train_data" / "audio"
    test_save_path = save_path / "test_data" / "audio"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    # data_root_list = [test_data_root, train_data_root]
    # save_path_list = [test_save_path, train_save_path]

    data_root_list = [test_data_root]
    save_path_list = [test_save_path]

    return data_root_list, mean_std_path, save_path_list


def get_path_test_vc(cfg, model_path, speaker, reference):
    if cfg.test.face_or_lip == "face":
        train_data_root = cfg.train.face_pre_loaded_path
        test_data_root = cfg.test.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    if cfg.test.face_or_lip == "lip":
        train_data_root = cfg.train.lip_pre_loaded_path
        test_data_root = cfg.test.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    
    train_data_root = Path(train_data_root).expanduser()
    test_data_root = Path(test_data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

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
    return data_root_list, mean_std_path, save_path_list


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
    # wandb.log({f"{filename}": plt})
    # wandb.log({f"Image {filename}": wandb.Image(os.path.join(save_path, f"{filename}.png"))})
    wandb.log({f"loss {filename}": wandb.plot.line_series(
        xs=np.arange(len(train_loss_list)), 
        ys=[train_loss_list, val_loss_list],
        keys=["train loss", "validation loss"],
        title=f"{filename}",
        xname="epoch",
    )})


def make_train_val_loader(cfg, data_root, mean_std_path):
    # パスを取得
    data_path = get_datasets(
        data_root=data_root,
        cfg=cfg,
    )
    data_path = random.sample(data_path, len(data_path))
    n_samples = len(data_path)
    train_size = int(n_samples * 0.95)
    train_data_path = data_path[:train_size]
    val_data_path = data_path[train_size:]

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
        mean_std_path = mean_std_path,
        transform=train_trans,
        cfg=cfg,
    )
    print("\n--- make validation dataset ---")
    val_dataset = KablabDataset(
        data_path=val_data_path,
        mean_std_path=mean_std_path,
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


def make_test_loader(cfg, data_root, mean_std_path):
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
        mean_std_path = mean_std_path,
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
        collate_fn=None
    )
  
    return test_loader, test_dataset


def check_mel_default(target, output, dec_output, cfg, filename, current_time, ckpt_time=None):
    tag = f'{current_time}_{cfg.tag}'
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
        save_path = save_path / cfg.train.name / tag
    else:
        save_path = save_path / cfg.train.name / tag
    print(save_path)
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