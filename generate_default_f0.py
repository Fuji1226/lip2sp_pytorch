import hydra

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from data_check import save_data, save_data_pwg
from train_default import make_model
from parallelwavegan.pwg_train import make_model as make_pwg
from calc_accuracy import calc_accuracy
from utils import make_test_loader_withf0, get_path_test, load_pretrained_model, gen_data_separate, gen_data_concat

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, pwg, test_loader, dataset, device, save_path):
    model.eval()
    pwg.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)

        lip_sep = gen_data_separate(lip, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        lip_len = lip_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output, dec_output, mixed_prev, fmaps, classifier_out, f0_pred = model(lip_sep, lip_len, spk_emb, f0_concat_target_or_pred="pred")

        output = gen_data_concat(
            output, 
            int(cfg.model.fps * cfg.model.reduction_factor), 
            int((lip_len[0] % cfg.model.fps) * cfg.model.reduction_factor)
        )

        _save_path = save_path / "griffinlim" / speaker[0] / filename[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        
        noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output.dtype)

        with torch.no_grad():
            wav_pred = pwg(noise, output)
            wav_abs = pwg(noise, feature)

        _save_path = save_path / "pwg" / speaker[0] / filename[0]
        os.makedirs(_save_path, exist_ok=True)

        save_data_pwg(
            cfg=cfg,
            save_path=_save_path,
            target=wav,
            output=wav_pred,
            ana_syn=wav_abs,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    pwg, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_300.ckpt").expanduser()
    pwg = load_pretrained_model(model_path_pwg, pwg, "gen")

    start_epoch = 20
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    cfg.model.use_lip_and_face = False
    cfg.model.use_f0_predicter = True
    model = make_model(cfg, device)
    
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:05:03_18-12-19/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss 
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:05:06_16-08-38/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss finetuning
        model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:05:09_08-48-21/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss finetuning only decoder

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader_withf0(cfg, data_root, train_data_root)

            print("--- generate ---")
            generate(
                cfg=cfg,
                model=model,
                pwg=pwg,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )
            
        for data_root, save_path in zip(data_root_list, save_path_list):
            for speaker in cfg.test.speaker:
                save_path_spk = save_path / "griffinlim" / speaker
                save_path_pwg_spk = save_path / "pwg" / speaker
                print("--- calc accuracy ---")
                calc_accuracy(save_path_spk, save_path.parents[0], cfg, "accuracy_griffinlim")
                calc_accuracy(save_path_pwg_spk, save_path.parents[0], cfg, "accuracy_pwg")

if __name__ == "__main__":
    main()