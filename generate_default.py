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
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature, load_pretrained_model
from data_process.phoneme_encode import get_keys_from_value

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)
    speaker_idx = dataset.speaker_idx

    input_length = cfg.model.n_lip_frames
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, spk_emb, speaker, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feat_add = feat_add.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        f0_target = feat_add[:, 0, :].unsqueeze(1)  # (B, 1, T)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep, landmark_sep, f0_target_sep = gen_separate(lip, input_length, shift_frame, f0_target, cfg.model.reduction_factor, landmark)
        data_len = data_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output, dec_output, mixed_prev, fmaps, f0, classifier_out = model(lip_sep, data_len, spk_emb)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / "griffinlim" / speaker_label / label[0]
        os.makedirs(_save_path, exist_ok=True)
        
        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        if f0 is not None:
            f0 = gen_cat_feature(f0, shift_frame, n_last_frame, upsample)
            plt.figure()
            plt.plot(f0_target.squeeze(0).squeeze(0))
            plt.plot(f0.squeeze(0).squeeze(0))
            plt.grid()
            plt.savefig(str(_save_path / "f0_comparison.png"))
            plt.close()

        # iter_cnt += 1
        # if iter_cnt == 53:
        #     break


def generate_pwg(cfg, model, gen, test_loader, dataset, device, save_path):
    model.eval()
    gen.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)
    speaker_idx = dataset.speaker_idx

    input_length = cfg.model.n_lip_frames
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, spk_emb, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        f0_target = feat_add[:, 0, :].unsqueeze(1)  # (B, 1, T)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep, landmark_sep, f0_target_sep = gen_separate(lip, input_length, shift_frame, f0_target, cfg.model.reduction_factor, landmark)
        data_len = data_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output, dec_output, mixed_prev, fmaps, f0, classifier_out = model(lip_sep, data_len, spk_emb)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output.dtype)
        wav_pred = gen(noise, output)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / "pwg" / speaker_label / label[0]
        os.makedirs(_save_path, exist_ok=True)
    
        save_data_pwg(
            cfg=cfg,
            save_path=_save_path,
            target=wav,
            output=wav_pred,
        )

        iter_cnt += 1
        if iter_cnt == 53:
            break


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    gen, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:06_08-35-21/mspec80_220.ckpt").expanduser()
    gen = load_pretrained_model(model_path_pwg, gen, "gen")

    start_epoch = 400
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        # glu
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_13-54-16/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_14-11-36/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_18-27-48/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:30_22-50-32/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss pitch
        model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2023:01:09_10-11-47/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss pitch train synth

        # glu no time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_16-36-04/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_16-28-46/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_17-06-45/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)
            
            generate(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )

            generate_pwg(
                cfg=cfg,
                model=model,
                gen=gen,
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