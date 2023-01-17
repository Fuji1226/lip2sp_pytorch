import hydra

from pathlib import Path
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from data_check import visualize_feature_map_video, visualize_feature_map_image
from train_nar import make_model
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature, gen_cat_wav, set_config, load_pretrained_model
from calc_accuracy import calc_accuracy
from data_process.phoneme_encode import get_keys_from_value

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


save_video = True
save_image = False


def generate(cfg, model, test_loader, dataset, device, save_path, mean_or_max):
    model.eval()
    speaker_idx = dataset.speaker_idx

    input_length = cfg.model.n_lip_frames
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, spk_emb, speaker, label = batch
        wav_q = wav_q.to(device)
        lip = lip.to(device)
        feature = feature.to(device)
        data_len = data_len.to(device)
        speaker = speaker.to(device)
        spk_emb = spk_emb.to(device)

        f0_target = feat_add[:, 0, :].unsqueeze(1)  # (B, 1, T)

        # n_last_frame = lip.shape[-1] % shift_frame
        # lip_sep, landmark_sep, f0_target_sep = gen_separate(lip, input_length, shift_frame, f0_target, cfg.model.reduction_factor, landmark)

        with torch.no_grad():
            output, classifier_out, fmaps = model(lip, landmark, data_len, spk_emb)

        # output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / f"fmaps_{mean_or_max}" / speaker_label / label[0]
        os.makedirs(_save_path, exist_ok=True)

        for fmap in fmaps[:1]:
            if save_video:
                visualize_feature_map_video(fmap, _save_path, mean_or_max)
            if save_image:
                visualize_feature_map_image(fmap, _save_path, mean_or_max)

        iter_cnt += 1
        if iter_cnt >= 3:
            break


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    mean_or_max = "mean"

    model = make_model(cfg, device)

    start_epoch = 400
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    for num_gen_epoch in num_gen_epoch_list:
        # single speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.3_50_gray/2022:12:09_13-29-45/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 03
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.8_50_gray/2022:12:09_13-46-31/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 08
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:09_14-02-12/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:12_10-27-44/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face delta
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:11_16-17-37/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face time masking
        
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:20_19-05-43/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face time masking scheduler
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:01:07_10-36-37/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F02 face time masking scheduler
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:30_07-49-41/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M01 face time masking scheduler
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:01:07_10-57-39/mspec80_{num_gen_epoch}.ckpt").expanduser()   # M04 face time masking scheduler

        # multi speaker
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.8_50_gray/2023:01:08_17-56-25/mspec80_{num_gen_epoch}.ckpt").expanduser()   # no emb lip 08
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:01:08_12-38-22/mspec80_{num_gen_epoch}.ckpt").expanduser()   # no emb face
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:27_01-08-42/mspec80_{num_gen_epoch}.ckpt").expanduser()   # no emb face time masking

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
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
                mean_or_max=mean_or_max,
            )
        

if __name__ == "__main__":
    main()