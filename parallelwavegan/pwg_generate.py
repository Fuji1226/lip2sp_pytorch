"""
F01のpwgはこれでいい
Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_300.ckpt").expanduser()
"""
import hydra

from pathlib import Path
import os
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import torch

from parallelwavegan.pwg_train import make_model
from data_check import save_data_pwg
from utils import make_test_loader_with_external_data, get_path_test, load_pretrained_model
from data_process.phoneme_encode import get_keys_from_value

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, gen, test_loader, dataset, device, save_path):
    gen.eval()

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        wav = wav.to(device).unsqueeze(1)
        feature = feature.to(device)

        noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
        wav_pred = gen(noise, feature)

        _save_path = save_path / speaker[0] / filename[0]
        os.makedirs(_save_path, exist_ok=True)

        save_data_pwg(
            cfg=cfg,
            save_path=_save_path,
            target=wav,
            output=wav_pred,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    gen, disc = make_model(cfg, device)

    start_epoch = 68
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        # model_path = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:06_08-35-21/mspec80_{num_gen_epoch}.ckpt").expanduser()
        # model_path = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:26_13-18-16/mspec80_{num_gen_epoch}.ckpt").expanduser()
        # model_path = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_{num_gen_epoch}.ckpt").expanduser()     # training 1 sec
        model_path = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:02:02_16-06-28/mspec80_{num_gen_epoch}.ckpt").expanduser()     # training 1 sec 

        gen = load_pretrained_model(model_path, gen, "gen")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)
        
        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader_with_external_data(cfg, data_root, train_data_root)
            generate(
                cfg=cfg,
                gen=gen,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )


if __name__ == "__main__":
    main()