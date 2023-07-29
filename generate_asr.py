import hydra
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import torch

from train_asr_amp import make_model
from utils import make_test_loader, get_path_test, load_pretrained_model
from jiwer import wer, cer

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, model, test_loader, dataset, device, save_path):
    per_list = []
    for batch in test_loader:
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        feature = feature.to(device)
        text = text.to(device)
        feature_len = feature_len.to(device)
        text_len = text_len.to(device)
        
        with torch.no_grad():
            output = model(feature, feature_len)

        text = text.squeeze(0).cpu().numpy().tolist()
        output = output.squeeze(0).cpu().numpy().tolist()
        text = [dataset.id_to_class[t] for t in text]
        output = [dataset.id_to_class[t] for t in output]
        text = ' '.join(text[1:-1])
        output = ' '.join(output[:-1])
        per = wer(text, output)
        per_list.append(per)

    per = np.mean(per_list)
    breakpoint()



@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    start_epoch = 20
    num_gen = 1
    num_gen_epoch_list = [start_epoch + int(i * 10) for i in range(num_gen)]

    model = make_model(cfg, device)
    for num_gen_epoch in num_gen_epoch_list:
        model_path = Path(f"~/lip2sp_pytorch/check_point/asr/face_cropped_max_size_fps25/2023:07:26_19-50-22/mspec80_{num_gen_epoch}.ckpt").expanduser()

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
            )


if __name__ == '__main__':
    main()