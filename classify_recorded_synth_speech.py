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
import re
import matplotlib.pyplot as plt

from train_recorded_synth_classifier_speech import make_model
from utils import load_pretrained_model, get_path_train, make_train_val_loader_tts

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    model = make_model(cfg, device)
    model_path = Path(f"~/lip2sp_pytorch/check_point/classifier/face_aligned_0_50_gray/2023:05:03_17-07-11/mspec80_49.ckpt").expanduser()   # F01 recorded and synth

    model = load_pretrained_model(model_path, model, "model")
    model.eval()
    cfg.train.face_or_lip = model_path.parents[1].name
    cfg.test.face_or_lip = model_path.parents[1].name

    recorded_path = Path(f"~/dataset/lip/np_files/face_aligned_0_50_gray/train").expanduser()
    synth_path = Path(f"~/dataset/lip/np_files_synth_corpus/face_aligned_0_50_gray/train").expanduser()
    save_path = Path("~/lip2sp_pytorch/result/classifier/classify").expanduser()
    save_path = save_path / model_path.parents[1].name / model_path.parents[0].name / model_path.stem
    
    speaker_list = ["F01_kablab"]
    path_all_dict = {}
    
    print("load recorded data")
    corpus_list = ["ATR", "BASIC5000", "balanced"]
    for speaker in speaker_list:
        spk_path_list = []
        spk_gt_path = recorded_path / speaker / cfg.model.name
        
        for corpus in corpus_list:
            spk_gt_path_co = [p for p in spk_gt_path.glob("*.npz") if re.search(f"{corpus}", str(p))]
            spk_path_list += spk_gt_path_co
        
        path_all_dict[f"{speaker}_recorded"] = spk_path_list
        
    print("load synthesized data")
    for speaker in speaker_list:
        spk_synth_path = synth_path / speaker / cfg.model.name
        spk_synth_path = list(spk_synth_path.glob("*.npz"))
        path_all_dict[f"{speaker}_synth"] = spk_synth_path
    
    print(f"load mean and std")
    train_data_root, val_data_root, ckpt_path, _, ckpt_time = get_path_train(cfg, current_time)
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_tts(cfg, train_data_root, val_data_root)
    feat_mean = train_dataset.feat_mean.to(device)
    feat_std = train_dataset.feat_std.to(device)
    feat_mean = feat_mean.unsqueeze(0)  # (1, C)
    feat_std = feat_std.unsqueeze(0)    # (1, C)
    
    print("classify")
    output_recorded_dict = {}
    output_synth_dict = {}
    for key, path_list in path_all_dict.items():
        print(key)
        speaker = "_".join(key.split("_")[:-1])
        condition = key.split("_")[-1]
        
        output_list = []
        for path in tqdm(path_list):
            npz_key = np.load(str(path))
            feature = torch.from_numpy(npz_key['feature']).to(device)      # (T, C)
            feature_len = torch.tensor([feature.shape[0]]).to(device)
            feature = (feature - feat_mean) / feat_std
            feature = feature.to(torch.float32).permute(1, 0).unsqueeze(0)  # (1, C, T)
            
            with torch.no_grad():
                output = model(feature, feature_len)
                output = torch.sigmoid(output).squeeze(1)   # (1,)
                
            output_list.append(output)
        
        output = torch.cat(output_list, dim=0).cpu().numpy()
        
        if condition == "recorded":
            output_recorded_dict[speaker] = output
        elif condition == "synth":
            output_synth_dict[speaker] = output
            
    for speaker in speaker_list:
        output_recorded = output_recorded_dict[speaker]
        output_synth = output_synth_dict[speaker]
        save_path_spk = save_path / speaker
        save_path_spk.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.hist(output_recorded, label="recorded", bins=40, color="c", alpha=0.5)
        plt.hist(output_synth, label="synthesized", bins=40, color="m", alpha=0.5)
        plt.legend()
        plt.grid()
        plt.savefig(str(save_path_spk / "output_histgram.png"))
        plt.close()
        
        
if __name__ == "__main__":
    main()