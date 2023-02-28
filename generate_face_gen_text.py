import hydra

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random
from tqdm import tqdm
import seaborn as sns

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from train_face_gen_text import make_model
from utils import load_pretrained_model, get_path_test_raw, make_test_loader_tts_face_raw
from data_check import save_data_face_gen

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

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label = batch
        text = text.to(device)
        lip = lip.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        lip_len = lip_len.to(device)
        spk_emb = spk_emb.to(device)

        with torch.no_grad():
            # dec_output, output, logit = model(text, text_len, lip_len, lip)
            dec_output, output, logit = model.inference(text, text_len)

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

    start_epoch = 10
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        model_path = Path(f"~/lip2sp_pytorch/check_point/face_gen_text/face/2023:02:26_09-12-06/mspec80_{num_gen_epoch}.ckpt").expanduser()

        model = load_pretrained_model(model_path, model, "model")
        cfg.train.face_or_lip = model_path.parents[1].name
        cfg.test.face_or_lip = model_path.parents[1].name

        data_dir, bbox_dir, landmark_dir, df_list, save_path_list, train_df = get_path_test_raw(cfg, model_path)

        for df, save_path in zip(df_list, save_path_list):
            test_loader, test_dataset = make_test_loader_tts_face_raw(data_dir, bbox_dir, landmark_dir, train_df, df, cfg)

            generate(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
            )