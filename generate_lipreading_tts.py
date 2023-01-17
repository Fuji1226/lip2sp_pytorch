from unicodedata import name
from omegaconf import DictConfig, OmegaConf
import hydra

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from train_lipreading import make_model as make_model_lipreading
from train_tts import make_model as make_model_tts
from parallelwavegan.pwg_train import make_model as make_pwg
from utils import get_path_test, make_test_loader_lipreading
from data_process.phoneme_encode import get_keys_from_value
from data_check import save_data_lipreading, save_data_tts, save_data_pwg
from generate_tts import check_attention_weight

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model_lipreading, model_tts, gen, test_loader, dataset, device, save_path):
    model_lipreading.eval()
    model_tts.eval()
    gen.eval()
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, text, data_len, text_len, speaker, label = batch

        lip = lip.to(device)
        text = text.to(device)
        data_len = data_len.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = text[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = text[:, 1:]    # (B, T)

        with torch.no_grad():
            output, ctc_output = model_lipreading(lip, n_max_loop=500, search_method="greedy")

            text = torch.ones(output.shape[0], output.shape[1] + 1).to(torch.long)
            text[:, 1:] = output
            text_len = torch.tensor([text.shape[1]])
            dec_output, output, logit, att_w = model_tts(text, text_len)

            noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output.dtype)
            wav_pred = gen(noise, output)

            noise = torch.randn(output.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
            wav_abs = gen(noise, feature)

        _save_path = save_path / speaker[0] / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        _save_path_gl = _save_path / "griffinlim"
        _save_path_gl.mkdir(parents=True, exist_ok=True)
        _save_path_pwg = _save_path / "pwg"
        _save_path_pwg.mkdir(parents=True, exist_ok=True)

        save_data_tts(
            cfg=cfg,
            save_path=_save_path_gl,
            wav=wav,
            feature=feature,
            output=output,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        check_attention_weight(att_w[0], cfg, "attention", _save_path)

        save_data_pwg(
            cfg=cfg,
            save_path=_save_path_pwg,
            target=wav,
            output=wav_pred,
            ana_syn=wav_abs,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # lipreading
    model_lipreading = make_model_lipreading(cfg, device)
    model_path_lipreading = Path(f"~/lip2sp_pytorch/check_point/lipreading/face_aligned_0_50_gray/2023:01:05_16-11-35/mspec80_200.ckpt").expanduser()  # F01
    cfg.train.face_or_lip = model_path_lipreading.parents[1].name
    cfg.test.face_or_lip = model_path_lipreading.parents[1].name

    if model_path_lipreading.suffix == ".ckpt":
        try:
            model_lipreading.load_state_dict(torch.load(str(model_path_lipreading))['model'])
        except:
            model_lipreading.load_state_dict(torch.load(str(model_path_lipreading), map_location=torch.device('cpu'))['model'])
    elif model_path_lipreading.suffix == ".pth":
        try:
            model_lipreading.load_state_dict(torch.load(str(model_path_lipreading)))
        except:
            model_lipreading.load_state_dict(torch.load(str(model_path_lipreading), map_location=torch.device('cpu')))        

    # tts
    model_tts = make_model_tts(cfg, device)
    model_path_tts = Path(f"~/lip2sp_pytorch/check_point/tts/face_aligned_0_50_gray/2023:01:08_10-33-05/mspec80_200.ckpt").expanduser()     # F01

    if model_path_tts.suffix == ".ckpt":
        try:
            model_tts.load_state_dict(torch.load(str(model_path_tts))['model'])
        except:
            model_tts.load_state_dict(torch.load(str(model_path_tts), map_location=torch.device('cpu'))['model'])
    elif model_path_tts.suffix == ".pth":
        try:
            model_tts.load_state_dict(torch.load(str(model_path_tts)))
        except:
            model_tts.load_state_dict(torch.load(str(model_path_tts), map_location=torch.device('cpu')))

    # parallelwavegan
    gen, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:06_08-35-21/mspec80_220.ckpt").expanduser()
    if model_path_pwg.suffix == ".ckpt":
        try:
            gen.load_state_dict(torch.load(str(model_path_pwg))['gen'])
        except:
            gen.load_state_dict(torch.load(str(model_path_pwg), map_location=torch.device('cpu'))['gen'])
    elif model_path_pwg.suffix == ".pth":
        try:
            gen.load_state_dict(torch.load(str(model_path_pwg)))
        except:
            gen.load_state_dict(torch.load(str(model_path_pwg), map_location=torch.device('cpu')))

    data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path_lipreading)

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader_lipreading(cfg, data_root, train_data_root)

        generate(
            cfg=cfg,
            model_lipreading=model_lipreading,
            model_tts=model_tts,
            gen=gen,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()