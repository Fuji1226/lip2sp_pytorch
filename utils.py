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
from collections import OrderedDict
import pandas as pd
import psutil
from collections import defaultdict

from dataset.dataset_npz import KablabDataset, KablabTransform, collate_time_adjust
from data_process.feature import wav2mel
from data_process.phoneme_encode import get_keys_from_value


def fix_random_seed(seed):
    """
    乱数固定
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def get_padding(kernel_size, dilation=1):
    """
    畳み込みおいてkernel_sizeにあったpaddingの値を計算
    """
    return (kernel_size*dilation - dilation) // 2


def get_path_train(cfg, current_time):
    """
    学習時のパス取得
    """
    # data
    if cfg.train.face_or_lip == "face_cropped_max_size_fps25":
        train_data_root = cfg.train.face_cropped_max_size_fps25_train
        val_data_root = cfg.train.face_cropped_max_size_fps25_val

    train_data_root = Path(train_data_root).expanduser()
    val_data_root = Path(val_data_root).expanduser()

    # 以前のcheckpointから開始する場合、そのディレクトリに追加で保存されるようにckpt_timeを取得
    ckpt_time = None
    if cfg.train.check_point_start:
        checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
        ckpt_time = checkpoint_path.parents[0].name

    # check point
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    if ckpt_time is not None:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / cfg.model.name / ckpt_time
    else:
        ckpt_path = ckpt_path / cfg.train.face_or_lip / cfg.model.name / current_time
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # save result
    save_path = Path(cfg.train.save_path).expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.face_or_lip / cfg.model.name / ckpt_time    
    else:
        save_path = save_path / cfg.train.face_or_lip / cfg.model.name / current_time
    save_path.mkdir(parents=True, exist_ok=True)

    return train_data_root, val_data_root, ckpt_path, save_path, ckpt_time


def get_path_test(cfg, model_path):
    """
    テスト時のパス取得
    """
    if cfg.train.face_or_lip == "face_cropped_max_size_fps25":
        train_data_root = cfg.train.face_cropped_max_size_fps25_train
        test_data_root = cfg.test.face_cropped_max_size_fps25_test
    
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


def exclude_unreadable_files(data_path_list):
    """
    読み込めないファイルがあれば除外
    """
    unreadable_data_path_list = []
    for data_path in data_path_list:
        try:
            npz_key = np.load(str(data_path))
        except:
            unreadable_data_path_list.append(data_path)
    data_path_list = list(set(data_path_list) - set(unreadable_data_path_list))
    return data_path_list


def get_datasets(data_root, cfg):
    """
    学習時にディレクトリからファイルのパス一覧を取得
    """
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
    items = exclude_unreadable_files(items)
    return items


def get_datasets_test(data_root, cfg):
    """
    テスト時にディレクトリからファイルのパス一覧を取得
    """
    print("\n--- get datasets ---")
    items = []
    for speaker in cfg.test.speaker:
        print(f"load {speaker}")
        spk_path = data_root / speaker / cfg.model.name
        spk_path = list(spk_path.glob("*.npz"))
        items += spk_path
    items = exclude_unreadable_files(items)
    return items


def make_train_val_loader(cfg, train_data_root, val_data_root):
    """
    学習用、検証用のDataLoaderを作成
    """
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

    # dataloader作成
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
        num_workers=0,      # 0じゃないとitoでバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_test_loader(cfg, data_root, train_data_root):
    """
    テスト用のDataLoaderを作成
    """
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


def count_params(module, attr):
    """
    モデルパラメータ数のカウント
    """
    params = 0
    for p in module.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"{attr}_parameter = {params}")


def requires_grad_change(net, val):
    """
    requires_gradの変更
    """
    for param in net.parameters():
        param.requires_grad = val
    return net


def save_loss(train_loss_list, val_loss_list, save_path, filename):
    """
    学習時の損失の保存
    """
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
    # model.load_state_dict(pretrained_dict)
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


def check_memory_status():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    used_memory = memory_info.used
    available_memory = memory_info.available
    memory_percent = memory_info.percent
    print(f"Total Memory: {total_memory} bytes")
    print(f"Used Memory: {used_memory} bytes")
    print(f"Available Memory: {available_memory} bytes")
    print(f"Memory Usage Percentage: {memory_percent}%")


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


def delete_unnecessary_checkpoint(result_dir, checkpoint_dir):
    """
    検証データに対しての最も小さくなったチェックポイント以外を削除
    （モデルの大きさによっては結構容量を取られるので）
    """
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