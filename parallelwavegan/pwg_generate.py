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
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature, gen_cat_wav, set_config
from data_process.phoneme_encode import get_keys_from_value

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, gen, test_loader, dataset, device, save_path):
    gen.eval()

    speaker_idx = dataset.speaker_idx

    process_times = []

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label = batch
        wav_q = wav_q.to(device)
        lip = lip.to(device)
        feature = feature.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=feature.dtype)
        wav_pred = gen(noise, feature)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / speaker_label / label[0]
        os.makedirs(_save_path, exist_ok=True)

        save_data_pwg(
            cfg=cfg,
            save_path=_save_path,
            target=wav,
            output=wav_pred,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    gen, disc = make_model(cfg, device)

    # gen 64
    # model_path = Path("~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2022:12:17_23-20-15/mspec80_500.ckpt").expanduser()
    # model_path = Path("~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2022:12:25_13-47-38/mspec80_60.ckpt").expanduser()
    model_path = Path("~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2022:12:30_21-57-55/mspec80_40.ckpt").expanduser()

    # gen 96
    # model_path = Path("~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2022:12:26_07-53-08/mspec80_500.ckpt").expanduser()
    
    cfg.train.face_or_lip = model_path.parents[1].name
    cfg.test.face_or_lip = model_path.parents[1].name

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

    data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)
    
    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)
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