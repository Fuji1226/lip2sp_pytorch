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

from data_check import save_data
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

    input_length = cfg.model.lip_min_frame
    shift_frame = input_length // 3

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, wav_q, lip, feature, feat_add, upsample, data_len, speaker, label = batch
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
                output, classifier_out = model(lip=lip_sep, gc=speaker)
            else:
                output, classifier_out = model(lip=lip_sep)

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

        iter_cnt += 1
        if iter_cnt >= 53:
            break
        
    return process_times


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)

    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/face_nn_gray_0_50/2022:11:22_00-30-18/mspec80_300.ckpt")   # F01 face
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/face_nn_gray_0_50/2022:11:28_02-55-07/mspec80_300.ckpt")   # F01 face crop flip
    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/face_nn_gray_0_50/2022:11:28_11-05-43/mspec80_300.ckpt")   # F01 face time masking
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/face_nn_0_50/2022:11:25_09-52-00/mspec80_300.ckpt")   # F01 face rgb
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_nn_gray_03_50/2022:11:22_01-09-23/mspec80_300.ckpt")   # F01 lip 03
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_nn_gray_08_50/2022:11:22_00-52-54/mspec80_300.ckpt")   # F01 lip 08
    
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:11_22-41-12/mspec80_400.ckpt")   # F02
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:11_22-06-16/mspec80_400.ckpt")   # M01
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:11_22-57-59/mspec80_400.ckpt")   # M04

    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:16_12-19-35/mspec80_400.ckpt")   # multi no spk_emb
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:16_15-07-32/mspec80_400.ckpt")   # multi spk_emb after enc
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:16_11-19-13/mspec80_400.ckpt")   # multi spk_emb classifier 0.05 after enc
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/nar/lip_st_gray_03/2022:11:16_00-32-31/mspec80_400.ckpt")   # multi spk_emb classifier 0.05 after res

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
        )

    for data_root, save_path in zip(data_root_list, save_path_list):
        for speaker in cfg.test.speaker:
            save_path_spk = save_path / speaker
            print("--- calc accuracy ---")
            calc_accuracy(save_path_spk, save_path.parents[0], cfg, process_times=None)
        

if __name__ == "__main__":
    main()