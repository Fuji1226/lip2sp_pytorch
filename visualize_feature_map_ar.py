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

from data_check import visualize_feature_map_video, visualize_feature_map_image
from train_default import make_model
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

save_video = True
save_image = True


def generate(cfg, model, test_loader, dataset, device, save_path, mean_or_max):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    input_length = cfg.model.n_lip_frames
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, landmark, feature_masked, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep = gen_separate(lip, input_length, shift_frame)
        
        start_time = time.time()

        with torch.no_grad():
            if cfg.train.use_gc:
                output, dec_output, mixed_prev, fmaps = model(lip=lip_sep, gc=speaker)
            else:
                output, dec_output, mixed_prev, fmaps = model(lip=lip_sep)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / label[0] / f"fmaps_{mean_or_max}"
        os.makedirs(_save_path, exist_ok=True)
        
        
        for fmap in fmaps:
            try:
                if save_video:
                    visualize_feature_map_video(fmap, _save_path, mean_or_max)
                if save_image:
                    visualize_feature_map_image(fmap, _save_path, mean_or_max)
            except:
                continue

        iter_cnt += 1
        if iter_cnt >= 5:
            break

    return process_times


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    mean_or_max = "mean"

    model = make_model(cfg, device)

    start_epoch = 360
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    for num_gen_epoch in num_gen_epoch_list:
        # glu
        model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_13-54-16/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_14-11-36/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:15_18-27-48/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking

        # glu no time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_16-36-04/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_16-28-46/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:17_17-06-45/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking

        # transformer
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:16_10-18-56/mspec80_{num_gen_epoch}.ckpt").expanduser()     # tf
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:16_07-29-34/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss
        # model_path = Path(f"~/lip2sp_pytorch/check_point/default/face_aligned_0_50_gray/2022:12:16_00-09-48/mspec80_{num_gen_epoch}.ckpt").expanduser()     # ss + masking


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

            process_times = None
            print("--- generate ---")
            process_times = generate(
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