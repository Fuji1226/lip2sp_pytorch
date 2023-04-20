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
from utils import make_test_loader, get_path_test, load_pretrained_model, gen_data_separate, gen_data_concat

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
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
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
            output, dec_output, mixed_prev, fmaps, classifier_out = model(lip_sep, lip_len, spk_emb)

        output = gen_data_concat(
            output, 
            int(cfg.model.fps * cfg.model.reduction_factor), 
            int((lip_len[0] % cfg.model.fps) * cfg.model.reduction_factor)
        )

        _save_path = save_path / "griffinlim" / speaker[0] / label[0]
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

        _save_path = save_path / "pwg" / speaker[0] / label[0]
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

    start_epoch = 500
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        # glu time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_13-54-16/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_14-11-36/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_18-27-48/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:02:01_23-53-31/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss pitch min 0.25 time masking
        
        # new
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:04:05_20-50-02/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:04:04_20-21-47/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:04:04_20-48-44/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss all

        # glu no time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_16-36-04/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_16-28-46/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_17-06-45/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:01:16_11-40-15/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss(scheduler変更後)
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:01:26_12-14-44/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf pitch 
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:02:01_23-35-54/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss pitch min 0.25

        # multi speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:01:20_11-36-09/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf no time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:01:20_11-36-09/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss time masking

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)

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