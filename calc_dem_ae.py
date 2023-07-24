import hydra

from pathlib import Path
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import librosa

import torch

from utils import make_test_loader, get_path_test
from train_ae_audio import make_model, calc_dem
    

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    vcnet, lip_enc = make_model(cfg, device)
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/ae/lip/2022:09:21_18-59-23/mspec80_200.ckpt")  # 0
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/ae/lip/2022:09:21_19-19-06/mspec80_200.ckpt")    # 0.1
    # model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/ae/lip/2022:09:22_12-06-45/mspec80_140.ckpt")    # 0.5
    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/ae/lip/2022:09:22_11-18-30/mspec80_140.ckpt")    # 1.0
    
    if model_path.suffix == ".ckpt":
        try:
            vcnet.load_state_dict(torch.load(str(model_path))['vcnet'])
        except:
            vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['vcnet'])
    elif model_path.suffix == ".pth":
        try:
            vcnet.load_state_dict(torch.load(str(model_path)))
        except:
            vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

    data_root_list, mean_std_path, save_path_list = get_path_test(cfg, model_path)

    for data_root, save_path in zip(data_root_list, save_path_list):
        cfg.test.speaker = ["F01_kablab"]
        test_loader_F01, test_dataset_F01 = make_test_loader(cfg, data_root, mean_std_path)
        test_loader_F01 = iter(test_loader_F01)

        cfg.test.speaker = ["M01_kablab"]
        test_loader_M01, test_dataset_M01 = make_test_loader(cfg, data_root, mean_std_path)
        test_loader_M01 = iter(test_loader_M01)

        dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01 = calc_dem(vcnet, test_loader_F01, test_loader_M01, cfg, device)
        with open(str(save_path.parents[0] / "dem.txt"), "a") as f:
            f.write("--- different speaker same utterance ---\n")
            f.write(f"speaker_embedding_dem = {dem_emb_both}, enc_output_dem = {dem_enc_both}\n")

            f.write("\n--- identical speaker different uttrance ---\n")
            f.write(f"F01_kablab\n")
            f.write(f"speaker_embedding_dem = {dem_emb_F01}, enc_output_dem = {dem_enc_F01}\n")

            f.write(f"\nM01_kablab\n")
            f.write(f"speaker_embedding_dem = {dem_emb_M01}, enc_output_dem = {dem_enc_M01}\n")



if __name__ == "__main__":
    main()