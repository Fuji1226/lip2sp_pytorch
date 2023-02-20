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

from parallelwavegan.pwg_train import make_model as make_pwg
from train_tts_raw import make_model
from utils import get_path_test, make_test_loader_tts, load_pretrained_model
from data_process.phoneme_encode import get_keys_from_value
from data_check import save_data_tts, save_data_pwg

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def check_attention_weight(att_w, cfg, filename, save_path):
    att_w = att_w.to('cpu').detach().numpy().copy()

    plt.figure()
    sns.heatmap(att_w, cmap="viridis", cbar=True)
    plt.title("attention weight")
    plt.xlabel("text")
    plt.ylabel("feature")

    plt.savefig(str(save_path / f"{filename}.png"))
    plt.close()


def generate(cfg, model, test_loader, dataset, device, save_path, gen):
    model.eval()
    gen.eval()
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    speaker_idx = dataset.speaker_idx

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, text, feature, stop_token, text_len, feature_len, spk_emb, speaker, label = batch
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)
        speaker = speaker.to(device)

        with torch.no_grad():
            if cfg.train.use_gc:
                dec_output, output, logit, att_w = model(text, text_len, feature_target=feature, spk_emb=spk_emb)
            else:
                dec_output, output, logit, att_w = model(text, text_len, feature_target=feature)

            noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output.dtype)
            wav_pred = gen(noise, output)

            noise = torch.randn(output.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
            wav_abs = gen(noise, feature)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / speaker_label / label[0]
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
    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    gen, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:06_08-35-21/mspec80_220.ckpt").expanduser()
    gen = load_pretrained_model(model_path_pwg, gen, "gen")

    start_epoch = 100
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        model_path = Path(f"~/lip2sp_pytorch/check_point/tts/face_aligned_0_50_gray/2023:01:08_10-33-05/mspec80_{num_gen_epoch}.ckpt").expanduser()     # F01

        # multi speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/tts/face_aligned_0_50_gray/2023:01:09_16-25-15/mspec80_{num_gen_epoch}.ckpt").expanduser()

        # women
        # model_path = Path(f"~/lip2sp_pytorch/check_point/tts/face_aligned_0_50_gray/2023:01:20_15-21-38/mspec80_{num_gen_epoch}.ckpt").expanduser()

        # men
        # model_path = Path(f"~/lip2sp_pytorch/check_point/tts/face_aligned_0_50_gray/2023:01:20_13-30-40/mspec80_{num_gen_epoch}.ckpt").expanduser()

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader_tts(cfg, data_root, train_data_root)
            
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