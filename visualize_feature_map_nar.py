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
from utils import make_test_loader, get_path_test, gen_separate, gen_cat_feature, gen_cat_wav, set_config
from calc_accuracy import calc_accuracy
from data_process.phoneme_encode import get_keys_from_value

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


save_video = False
save_image = True


def generate(cfg, model, test_loader, dataset, device, save_path, mean_or_max):
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
        data_len = data_len.to(device)
        speaker = speaker.to(device)

        n_last_frame = lip.shape[-1] % shift_frame
        lip_sep = gen_separate(lip, input_length, shift_frame)

        start_time = time.time()

        with torch.no_grad():
            if cfg.train.use_gc:
                output, classifier_out, fmaps = model(lip=lip, gc=speaker)
            else:
                output, classifier_out, fmaps = model(lip=lip)

        output = gen_cat_feature(output, shift_frame, n_last_frame, upsample)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        speaker_label = get_keys_from_value(speaker_idx, speaker[0])
        _save_path = save_path / speaker_label / label[0] / f"fmaps_{mean_or_max}"
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
    set_config(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    mean_or_max = "mean"

    model = make_model(cfg, device)

    start_epoch = 300
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    for num_gen_epoch in num_gen_epoch_list:
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:09_14-02-12/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2022:12:11_16-17-37/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 face time masking
        # model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.3_50_gray/2022:12:09_13-29-45/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 03
        model_path = Path(f"~/lip2sp_pytorch/check_point/nar/lip_cropped_0.8_50_gray/2022:12:09_13-46-31/mspec80_{num_gen_epoch}.ckpt").expanduser()   # F01 lip 08

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