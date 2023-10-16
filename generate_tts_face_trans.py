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
import seaborn as sns
from torch.cuda.amp import autocast

from parallelwavegan.pwg_train import make_model as make_pwg
from train_tts_face_trans_ddp import make_model
from utils import get_path_test, make_test_loader_tts_face, load_pretrained_model
from data_process.phoneme_encode import get_keys_from_value
from data_check import save_data_tts, save_data_pwg, save_data_face_gen

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, test_loader, dataset, device, save_path, gen):
    model.eval()
    gen.eval()
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    speaker_idx = dataset.speaker_idx

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label = batch
        text = text.to(device)
        feature = feature.to(device)
        lip = lip.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)
        lip_len = lip_len.to(device)
        spk_emb = spk_emb.to(device)

        with autocast():
            with torch.no_grad():
                dec_feat_output, dec_lip_output, feat_output, lip_output = model.inference(text, text_len, lip_len, spk_emb=spk_emb)

                noise = torch.randn(feat_output.shape[0], 1, feat_output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feat_output.dtype)
                wav_pred = gen(noise, feat_output)

                noise = torch.randn(feat_output.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
                wav_abs = gen(noise, feature)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / speaker_label / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        _save_path_gl = _save_path / "griffinlim"
        _save_path_gl.mkdir(parents=True, exist_ok=True)
        _save_path_pwg = _save_path / "pwg"
        _save_path_pwg.mkdir(parents=True, exist_ok=True)
        _save_path_face_gen = _save_path / "face_gen"
        _save_path_face_gen.mkdir(parents=True, exist_ok=True)

        feat_output = feat_output.to(torch.float32)
        lip_output = lip_output.to(torch.float32)
        wav_pred = wav_pred.to(torch.float32)
        wav_abs = wav_abs.to(torch.float32)
        feat_output = feat_output.to(torch.float32)

        save_data_tts(
            cfg=cfg,
            save_path=_save_path_gl,
            wav=wav,
            feature=feature,
            output=feat_output,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        save_data_pwg(
            cfg=cfg,
            save_path=_save_path_pwg,
            target=wav,
            output=wav_pred,
            ana_syn=wav_abs,
        )

        save_data_face_gen(
            cfg=cfg,
            save_path=_save_path_face_gen,
            wav=wav,
            target=lip,
            output=lip_output,
            lip_mean=lip_mean,
            lip_std=lip_std,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    gen, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_300.ckpt").expanduser()
    gen = load_pretrained_model(model_path_pwg, gen, "gen")

    start_epoch = 169
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg).to(device)
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        model_path = Path(f"~/lip2sp_pytorch/check_point/tts_face_trans/face_aligned_0_50_gray/2023:02:10_01-10-12/mspec80_{num_gen_epoch}.ckpt").expanduser()     # F01 

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader_tts_face(cfg, data_root, train_data_root)
            generate(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
                gen=gen,
            )


if __name__ == "__main__":
    main()