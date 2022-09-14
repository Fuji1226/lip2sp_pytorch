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
from torch.utils.data import DataLoader

from dataset.dataset_npz import KablabDataset, KablabTransform, get_datasets
from data_check import save_data
from train_vq_audio import make_model
from calc_dem_ae import get_path, make_test_loader


def calc_dem(model, loader_F01, loader_M01, cfg, device):
    spk_emb_f_list = []
    spk_emb_m_list = []
    quantize_f_list = []
    quantize_m_list = []

    print("--- model processing ---")
    for _ in tqdm(range(len(loader_F01))):
        wav_f, lip_f, feature_f, feat_add_f, upsample_f, data_len_f, speaker_f, label_f = loader_F01.next()
        wav_m, lip_m, feature_m, feat_add_m, upsample_m, data_len_m, speaker_m, label_m = loader_M01.next()
        
        with torch.no_grad():
            output_f, feat_add_out_f, phoneme_f, spk_emb_f, quantize_f, embed_idx_f, vq_loss_f, enc_output_f, idx_pred_f, spk_class_f = model(feature=feature_f, feature_ref=feature_m)
            output_m, feat_add_out_m, phoneme_m, spk_emb_m, quantize_m, embed_idx_m, vq_loss_m, enc_output_m, idx_pred_m, spk_class_m = model(feature=feature_m, feature_ref=feature_f)

        spk_emb_f = spk_emb_f[0].to('cpu').detach().numpy().copy()    # (C,)
        spk_emb_m = spk_emb_m[0].to('cpu').detach().numpy().copy()    # (C,)
        quantize_f = quantize_f[0].to('cpu').detach().numpy().copy()    # (T, C)
        quantize_m = quantize_m[0].to('cpu').detach().numpy().copy()    # (T, C)

        spk_emb_f_list.append(spk_emb_f)
        spk_emb_m_list.append(spk_emb_m)
        quantize_f_list.append(quantize_f)
        quantize_m_list.append(quantize_m)

    spk_emb_f_list_shuffle = random.sample(spk_emb_f_list, len(spk_emb_f_list))
    spk_emb_m_list_shuffle = random.sample(spk_emb_m_list, len(spk_emb_m_list))
    quantize_f_list_shuffle = random.sample(quantize_f_list, len(quantize_f_list))
    quantize_m_list_shuffle = random.sample(quantize_m_list, len(quantize_m_list))

    # 異なる話者で同じ発話内容の場合
    print("--- different speaker same utterance ---")
    cos_sim_emb_list = []
    cos_sim_enc_list = []
    for emb_f, emb_m, enc_f, enc_m in tqdm(zip(spk_emb_f_list, spk_emb_m_list, quantize_f_list, quantize_m_list)):
        # spk_emb
        cos_sim_emb = np.dot(emb_f, emb_m) / (np.linalg.norm(emb_f) * np.linalg.norm(emb_m))
        cos_sim_emb_list.append(cos_sim_emb)

        # enc_output
        min_cost, wp = librosa.sequence.dtw(enc_f.T, enc_m.T)
        cos_sim_enc_frame = []
        for i in range(wp.shape[0]):
            f_frame = enc_f[wp[i, 0]]
            m_frame = enc_m[wp[i, 1]]
            cos_sim_enc = np.dot(f_frame, m_frame) / (np.linalg.norm(f_frame) * np.linalg.norm(m_frame))
            cos_sim_enc_frame.append(cos_sim_enc)
        cos_sim_enc = sum(cos_sim_enc_frame) / len(cos_sim_enc_frame)
        cos_sim_enc_list.append(cos_sim_enc)

    dem_emb_both = sum(cos_sim_emb_list) / len(cos_sim_emb_list)
    dem_enc_both = sum(cos_sim_enc_list) / len(cos_sim_enc_list)

    # 同じ話者で異なる発話内容の場合
    print("--- identical speaker different utterance ---")
    print("F01_kablab")
    # F01_kablab
    cos_sim_emb_list = []
    cos_sim_enc_list = []
    for emb, emb_shuffle, enc, enc_shuffle in tqdm(zip(spk_emb_f_list, spk_emb_f_list_shuffle, quantize_f_list, quantize_f_list_shuffle)):
        # spk_emb
        cos_sim_emb = np.dot(emb, emb_shuffle) / (np.linalg.norm(emb) * np.linalg.norm(emb_shuffle))
        cos_sim_emb_list.append(cos_sim_emb)        

        # enc_output
        min_cost, wp = librosa.sequence.dtw(enc.T, enc_shuffle.T)
        cos_sim_enc_frame = []
        for i in range(wp.shape[0]):
            frame = enc[wp[i, 0]]
            frame_shuffle = enc_shuffle[wp[i, 1]]
            cos_sim_enc = np.dot(frame, frame_shuffle) / (np.linalg.norm(frame) * np.linalg.norm(frame_shuffle))
            cos_sim_enc_frame.append(cos_sim_enc)
        cos_sim_enc = sum(cos_sim_enc_frame) / len(cos_sim_enc_frame)
        cos_sim_enc_list.append(cos_sim_enc)
    
    dem_emb_F01 = sum(cos_sim_emb_list) / len(cos_sim_emb_list)
    dem_enc_F01 = sum(cos_sim_enc_list) / len(cos_sim_enc_list)

    # M01_kablab
    print("M01_kablab")
    cos_sim_emb_list = []
    cos_sim_enc_list = []
    for emb, emb_shuffle, enc, enc_shuffle in tqdm(zip(spk_emb_m_list, spk_emb_m_list_shuffle, quantize_m_list, quantize_m_list_shuffle)):
        # spk_emb
        cos_sim_emb = np.dot(emb, emb_shuffle) / (np.linalg.norm(emb) * np.linalg.norm(emb_shuffle))
        cos_sim_emb_list.append(cos_sim_emb)        

        # enc_output
        min_cost, wp = librosa.sequence.dtw(enc.T, enc_shuffle.T)
        cos_sim_enc_frame = []
        for i in range(wp.shape[0]):
            frame = enc[wp[i, 0]]
            frame_shuffle = enc_shuffle[wp[i, 1]]
            cos_sim_enc = np.dot(frame, frame_shuffle) / (np.linalg.norm(frame) * np.linalg.norm(frame_shuffle))
            cos_sim_enc_frame.append(cos_sim_enc)
        cos_sim_enc = sum(cos_sim_enc_frame) / len(cos_sim_enc_frame)
        cos_sim_enc_list.append(cos_sim_enc)
    
    dem_emb_M01 = sum(cos_sim_emb_list) / len(cos_sim_emb_list)
    dem_enc_M01 = sum(cos_sim_enc_list) / len(cos_sim_enc_list)
    return dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    model = make_model(cfg, device)
    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/vq/lip/2022:09:13_16-55-55/mspec80_200.ckpt") 
    
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

    data_root_list, mean_std_path, save_path_list = get_path(cfg, model_path)

    for data_root, save_path in zip(data_root_list, save_path_list):
        cfg.train.speaker = ["F01_kablab"]
        test_loader_F01, test_dataset_F01 = make_test_loader(cfg, data_root, mean_std_path)
        test_loader_F01 = iter(test_loader_F01)

        cfg.train.speaker = ["M01_kablab"]
        test_loader_M01, test_dataset_M01 = make_test_loader(cfg, data_root, mean_std_path)
        test_loader_M01 = iter(test_loader_M01)

        dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01 = calc_dem(model, test_loader_F01, test_loader_M01, cfg, device)
        with open(str(save_path / "dem.txt"), "a") as f:
            f.write("--- different speaker same utterance ---\n")
            f.write(f"speaker_embedding_dem = {dem_emb_both}, enc_output_dem = {dem_enc_both}\n")

            f.write("\n--- identical speaker different uttrance ---\n")
            f.write(f"F01_kablab\n")
            f.write(f"speaker_embedding_dem = {dem_emb_F01}, enc_output_dem = {dem_enc_F01}\n")

            f.write(f"\nM01_kablab\n")
            f.write(f"speaker_embedding_dem = {dem_emb_M01}, enc_output_dem = {dem_enc_M01}\n")


if __name__ == "__main__":
    main()