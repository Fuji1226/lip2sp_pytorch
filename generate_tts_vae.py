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
from train_tts_vae import make_model
from utils import get_path_test, make_test_loader, load_pretrained_model, get_path_train, make_train_val_loader_tts
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


def generate(cfg, model, test_loader, loader_for_vae, dataset, device, save_path, pwg):
    model.eval()
    pwg.eval()
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        batch_train = loader_for_vae.next()
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch_train
        feature_for_vae = feature.to(device)
        feature_for_vae_len = feature_len.to(device)
        
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)

        with torch.no_grad():
            dec_output, output, logit, att_w, mu, logvar = model(text, text_len, feature_for_vae, feature_for_vae_len, feature_target=None, spk_emb=spk_emb)

            noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output.dtype)
            wav_pred = pwg(noise, output)

            noise = torch.randn(output.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
            wav_abs = pwg(noise, feature)

        _save_path_gl = save_path / "griffinlim" / speaker[0] / label[0]
        _save_path_gl.mkdir(parents=True, exist_ok=True)
        _save_path_pwg = save_path / "pwg" / speaker[0] / label[0]
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
        check_attention_weight(att_w[0], cfg, "attention", _save_path_gl)
        save_data_pwg(
            cfg=cfg,
            save_path=_save_path_pwg,
            target=wav,
            output=wav_pred,
            ana_syn=wav_abs,
        )
        check_attention_weight(att_w[0], cfg, "attention", _save_path_pwg)
        

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    pwg, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_300.ckpt").expanduser()
    pwg = load_pretrained_model(model_path_pwg, pwg, "gen")

    start_epoch = 200
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        model_path = Path(f"~/lip2sp_pytorch/check_point/tts_vae/face_aligned_0_50_gray/2023:04:09_11-09-23/mspec80_{num_gen_epoch}.ckpt").expanduser()     # F01

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name
        cfg.train.batch_size = 1
        
        train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)
            
            train_loader, val_loader, _, _ = make_train_val_loader_tts(cfg, train_data_root, val_data_root)
            loader_for_vae = iter(train_loader)
            
            generate(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                loader_for_vae=loader_for_vae,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
                pwg=pwg,
            )


if __name__ == "__main__":
    main()