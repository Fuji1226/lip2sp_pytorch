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

from train_face_gen import make_model
from utils import make_test_loader, get_path_test, load_pretrained_model, get_path_train, make_train_val_loader,\
    gen_data_separate, gen_data_concat
from data_check import save_data_face_gen

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, gen, test_loader, dataset, loader_for_first_frame,  device, save_path):
    gen.eval()
    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        batch_train = loader_for_first_frame.next()
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch_train
        lip_first_frame = lip[..., 0]
        lip_first_frame = lip_first_frame.to(device)

        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        
        feature_sep = gen_data_separate(
            feature, 
            int(cfg.model.input_lip_sec * cfg.model.sampling_rate // cfg.model.hop_length), 
            cfg.model.sampling_rate // cfg.model.hop_length,
        )
        lip_len = feature_len // ((cfg.model.sampling_rate // cfg.model.hop_length) // cfg.model.fps)
        lip_len = lip_len.expand(feature_sep.shape[0])
        lip_first_frame = lip_first_frame.expand(feature_sep.shape[0], -1, -1, -1)

        with torch.no_grad():
            output = gen(lip_first_frame, feature_sep, lip_len)
        
        output = gen_data_concat(output, cfg.model.fps, lip_len[0] % cfg.model.fps)

        _save_path = save_path / speaker[0] / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)

        save_data_face_gen(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            target=lip,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
        )
        print("saved\n")


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    start_epoch = 100
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    gen, _, _, _, _ = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        model_path = Path(f"~/lip2sp_pytorch/check_point/face_gen/face_aligned_0_50_gray/2023:04:10_13-15-28/mspec80_{num_gen_epoch}.ckpt").expanduser()

        gen = load_pretrained_model(model_path, gen, "gen")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name
        cfg.train.batch_size = 1

        train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
        data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)

        for data_root, save_path in zip(data_root_list, save_path_list):
            test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)
            
            train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader(cfg, train_data_root, val_data_root)
            loader_for_first_frame = iter(train_loader)

            print("--- generate ---")
            generate(
                cfg=cfg,
                gen=gen,
                test_loader=test_loader,
                dataset=test_dataset,
                loader_for_first_frame=loader_for_first_frame,
                device=device,
                save_path=save_path,
            )


if __name__ == "__main__":
    main()