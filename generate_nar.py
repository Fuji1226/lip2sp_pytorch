import hydra

from pathlib import Path
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import torch

from data_check import save_data, save_data_pwg
from train_nar import make_model
from parallelwavegan.pwg_train import make_model as make_pwg
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature, gen_cat_wav, set_config
from calc_accuracy import calc_accuracy
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

    process_times = []

    input_length = cfg.model.n_lip_frames
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label = batch
        wav_q = wav_q.to(device)
        lip = lip.to(device)
        feature = feature.to(device)
        feat_add = feat_add.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        f0_target = feat_add[:, 0, :].unsqueeze(1)  # (B, 1, T)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep, landmark_sep, f0_target_sep = gen_separate(lip, input_length, shift_frame, f0_target, cfg.model.reduction_factor, landmark)

        start_time = time.time()

        with torch.no_grad():
            if cfg.train.use_gc:
                output, classifier_out, fmaps = model(lip=lip_sep, gc=speaker)
            else:
                output, classifier_out, fmaps = model(lip=lip_sep, landmark=landmark_sep)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / speaker_label / label[0]
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

        # iter_cnt += 1
        # if iter_cnt >= 53:
        #     break
        
    return process_times


def generate_pwg(cfg, model, gen, test_loader, dataset, device, save_path):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)
    speaker_idx = dataset.speaker_idx

    process_times = []

    input_length = cfg.model.n_lip_frames
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label = batch
        wav_q = wav_q.to(device)
        lip = lip.to(device)
        feature = feature.to(device)
        feat_add = feat_add.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        f0_target = feat_add[:, 0, :].unsqueeze(1)  # (B, 1, T)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep, landmark_sep, f0_target_sep = gen_separate(lip, input_length, shift_frame, f0_target, cfg.model.reduction_factor, landmark)

        start_time = time.time()

        with torch.no_grad():
            if cfg.train.use_gc:
                output, classifier_out, fmaps = model(lip=lip_sep, gc=speaker)
            else:
                output, classifier_out, fmaps = model(lip=lip_sep, landmark=landmark_sep)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        noise = torch.randn(output.shape[0], 1, output.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output.dtype)
        wav_pred = gen(noise, output)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / "pwg" / speaker_label / label[0]
        os.makedirs(_save_path, exist_ok=True)

        # save_data(
        #     cfg=cfg,
        #     save_path=_save_path,
        #     wav=wav,
        #     lip=lip,
        #     feature=feature,
        #     feat_add=feat_add,
        #     output=output,
        #     lip_mean=lip_mean,
        #     lip_std=lip_std,
        #     feat_mean=feat_mean,
        #     feat_std=feat_std,
        # )

        save_data_pwg(
            cfg=cfg,
            save_path=_save_path,
            target=wav,
            output=wav_pred,
        )

        # iter_cnt += 1
        # if iter_cnt >= 53:
        #     break


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)

    gen, disc = make_pwg(cfg, device)
    model_path = Path("~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2022:12:30_21-57-55/mspec80_50.ckpt").expanduser()
    if model_path.suffix == ".ckpt":
        try:
            gen.load_state_dict(torch.load(str(model_path))['gen'])
        except:
            gen.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['gen'])
    elif model_path.suffix == ".pth":
        try:
            gen.load_state_dict(torch.load(str(model_path)))
        except:
            gen.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

    start_epoch = 400
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    for num_gen_epoch in num_gen_epoch_list:
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.3_50_gray/2022:12:09_13-29-45/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 03
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.8_50_gray/2022:12:09_13-46-31/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 08
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:09_14-02-12/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:12_10-27-44/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face delta
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:11_16-17-37/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face time masking
        
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:20_19-05-43/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face time masking scheduler
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:20_19-05-43/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F02 face time masking scheduler
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:20_19-05-43/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M01 face time masking scheduler
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:20_19-05-43/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M04 face time masking scheduler

        # multi speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:27_01-08-42/mspec80_{num_gen_epoch}.ckpt").expanduser()   # no emb

        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        if model_path.suffix == ".ckpt":
            try:
                model.load_state_dict(torch.load(str(model_path))['model'])
            except:
                model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['model'])
        elif model_path.suffix == ".pth":
            try:
                model.load_state_dict(torch.load(str(model_path)))
            except:
                model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)
        
        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)

            generate_pwg(
                cfg=cfg,
                model=model,
                gen=gen,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )

            process_times = None
            print("--- generate ---")
            process_times = generate(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )

        for data_root, save_path in zip(data_root_list, save_path_list):
            for speaker in cfg.test.speaker:
                save_path_spk = save_path / speaker
                print("--- calc accuracy ---")
                calc_accuracy(save_path_spk, save_path.parents[0], cfg, process_times=None)
        

if __name__ == "__main__":
    main()