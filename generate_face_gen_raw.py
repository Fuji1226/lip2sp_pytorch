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

from train_face_gen_raw import make_model
from utils import make_test_loader_face_gen_raw, get_path_test_raw, load_pretrained_model, gen_data_separate, gen_data_concat
from data_check import save_data_face_gen

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, gen, test_loader, dataset, train_loader,  device, save_path):
    gen.eval()
    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        batch_train = train_loader.next()
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, label = batch_train
        lip_first_frame = lip[..., 0]
        lip_first_frame = lip_first_frame.to(device)

        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, label = batch
        lip = lip.to(device)
        feature = feature.to(device)

        feature_sep = gen_data_separate(
            feature, 
            int(cfg.model.input_lip_sec * cfg.model.fps * cfg.model.reduction_factor), 
            int(cfg.model.fps * cfg.model.reduction_factor)
        )

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


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    start_epoch = 110
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    gen, _, _, _, _ = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        model_path = Path(f"~/lip2sp_pytorch/check_point/face_gen/face/2023:02:24_16-27-37/mspec80_{num_gen_epoch}.ckpt").expanduser()

        gen = load_pretrained_model(model_path, gen, "gen")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_dir, bbox_dir, landmark_dir, df_list, save_path_list, train_df = get_path_test_raw(cfg, model_path)

        for df, save_path in zip(df_list, save_path_list):
            test_loader, test_dataset, train_loader, train_dataset = make_test_loader_face_gen_raw(data_dir, bbox_dir, landmark_dir, train_df, df, cfg)
            train_loader = iter(train_loader)

            print("--- generate ---")
            generate(
                cfg=cfg,
                gen=gen,
                test_loader=test_loader,
                dataset=test_dataset,
                train_loader=train_loader,
                device=device,
                save_path=save_path,
            )


if __name__ == "__main__":
    main()