import hydra
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
from utils import make_test_loader, get_path_test, load_pretrained_model, gen_data_separate, gen_data_concat
from calc_accuracy import calc_accuracy

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

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)

        lip_sep = gen_data_separate(lip, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        lip_len = lip_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output, classifier_out, fmaps = model(lip_sep, lip_len, spk_emb)

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

    start_epoch = 400
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.8_50_gray/2023:04:05_20-39-02/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 08
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:04:05_20-50-45/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:04:04_17-15-09/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:04:04_20-48-17/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01_all face time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:04:30_10-50-07/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 recorded and synth
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:06:04_01-13-00/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 gan 0.01
        
        # single speaker atr only
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:05:29_11-46-23/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:05:29_12-05-50/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F02 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:05:30_01-11-26/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M01 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:05:30_01-48-07/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M04 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:09_11-40-03/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:09_12-32-22/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F02 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:09_12-53-41/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M01 atr
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:09_14-58-39/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M04 atr
        
        # multi speaker atr only
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:15_01-19-13/mspec80_{num_gen_epoch}.ckpt").expanduser()   # simple
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:09_10-59-45/mspec80_{num_gen_epoch}.ckpt").expanduser()   # simple
        
        # multi speaker all
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:02_12-46-05/mspec80_{num_gen_epoch}.ckpt").expanduser()   # simple
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:02_12-01-35/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.1 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:11_00-38-10/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.05 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:11_07-32-39/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.025 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:02_20-47-43/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.01 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:11_10-04-40/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.1 enc
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size/2023:06:11_13-38-33/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.05 enc
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:07_09-28-20/mspec80_{num_gen_epoch}.ckpt").expanduser()   # simple
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:19_00-32-50/mspec80_{num_gen_epoch}.ckpt").expanduser()   # simple local
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:26_02-00-52/mspec80_{num_gen_epoch}.ckpt").expanduser()   # simple local conformer
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:07_17-16-01/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.1 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:08_12-29-25/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.05 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:08_14-15-16/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.025 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:08_20-01-41/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.01 res
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:08_21-33-06/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.1 enc
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:09_00-19-59/mspec80_{num_gen_epoch}.ckpt").expanduser()   # adversarial 0.05 enc
        
        # finetuning lr=0.0001 japanese normalization
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:24_01-43-48/mspec80_{num_gen_epoch}.ckpt").expanduser()    # all
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:23_01-08-23/mspec80_{num_gen_epoch}.ckpt").expanduser()    # fix res
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:23_02-01-37/mspec80_{num_gen_epoch}.ckpt").expanduser()    # fix enc
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:23_03-44-37/mspec80_{num_gen_epoch}.ckpt").expanduser()    # fix dec
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:23_05-26-21/mspec80_{num_gen_epoch}.ckpt").expanduser()    # fix res enc
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:23_06-19-34/mspec80_{num_gen_epoch}.ckpt").expanduser()    # fix res dec
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25/2023:07:23_07-12-28/mspec80_{num_gen_epoch}.ckpt").expanduser()    # fix enc dec
        
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