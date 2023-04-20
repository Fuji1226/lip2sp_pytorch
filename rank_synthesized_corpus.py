import hydra
from pathlib import Path
import numpy as np
import torch
import random
import re
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from train_tts_vae import make_model as make_tts
from utils import load_pretrained_model, get_path_train, make_train_val_loader_tts
from model.rank_svm import RankSVM

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tts = make_tts(cfg, device)
    # tts_path = Path(f"~/lip2sp_pytorch/check_point/tts_vae/face_aligned_0_50_gray/2023:04:09_11-09-23/mspec80_200.ckpt").expanduser()
    tts_path = Path(f"~/lip2sp_pytorch/check_point/tts_vae/face_aligned_0_50_gray/2023:04:20_02-35-04/mspec80_5.ckpt").expanduser()   # after finetuning
    tts = load_pretrained_model(tts_path, tts, "model")
    
    cfg.train.face_or_lip = tts_path.parents[1].name
    cfg.test.face_or_lip = tts_path.parents[1].name
    cfg.train.batch_size = 1
    
    recorded_path = Path(f"~/dataset/lip/np_files/{cfg.train.face_or_lip}/train").expanduser()
    synth_path = Path(f"~/dataset/lip/np_files_synth_corpus/{cfg.train.face_or_lip}/train").expanduser()
    
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
    
    print("calc vae feature")
    vae_feature_recorded_dict = {}
    vae_feature_synth_dict = {}
    for key, path_list in path_all_dict.items():
        print(key)
        speaker = "_".join(key.split("_")[:-1])
        condition = key.split("_")[-1]
        
        vae_feature_list = []
        for path in tqdm(path_list):
            npz_key = np.load(str(path))
            feature = torch.from_numpy(npz_key['feature']).to(device)      # (T, C)
            feature_len = torch.tensor([feature.shape[0]]).to(device)
            feature = (feature - feat_mean) / feat_std
            feature = feature.to(torch.float32).permute(1, 0).unsqueeze(0)  # (1, C, T)
            
            with torch.no_grad():
                mu, logvar, z, emb = tts.vae(feature, feature_len)
                
            std = torch.exp(0.5 * logvar)
            vae_feature = torch.cat((mu, std), dim=-1)   # (1, C)
            vae_feature_list.append(vae_feature)
            
        vae_feature = torch.cat(vae_feature_list, dim=0).cpu().numpy()    # (N, C)
        
        if condition == "recorded":
            vae_feature_recorded_dict[speaker] = vae_feature
        elif condition == "synth":
            vae_feature_synth_dict[speaker] = vae_feature
            
    print("ranksvm training")
    ranksvm_dict = {}
    for speaker in speaker_list:
        print(speaker)
        ranksvm = RankSVM(n_iter=100000)
        vae_feature_recorded = vae_feature_recorded_dict[speaker]
        vae_feature_synth = vae_feature_synth_dict[speaker]
        ranksvm.fit(vae_feature_recorded, vae_feature_synth)
        ranksvm_dict[speaker] = ranksvm
        
    print("ranksvm prediction")
    save_path = Path("~/lip2sp_pytorch/result/tts_vae/generate").expanduser()
    save_path = save_path / tts_path.parents[1].name / tts_path.parents[0].name / tts_path.stem / "test_data" / "ranking_result"
    
    for speaker in speaker_list:
        print(speaker)
        ranksvm = ranksvm_dict[speaker]
        vae_feature_recorded = vae_feature_recorded_dict[speaker]
        vae_feature_synth = vae_feature_synth_dict[speaker]
        
        pred_recorded_list = []
        pred_synth_list = []
        for i in range(vae_feature_recorded.shape[0]):
            pred = ranksvm.predict(vae_feature_recorded[i])
            pred_recorded_list.append(pred)
        for i in range(vae_feature_synth.shape[0]):
            pred = ranksvm.predict(vae_feature_synth[i])
            pred_synth_list.append(pred)
            
        pred_max = pred_recorded_list + pred_synth_list
        pred_max = np.max(np.abs(pred_max))
        pred_recorded_list = np.array(pred_recorded_list)
        pred_synth_list = np.array(pred_synth_list)
        pred_recorded_list /= pred_max
        pred_synth_list /= pred_max
        
        save_path_spk = save_path / speaker
        save_path_spk.mkdir(parents=True, exist_ok=True)
        
        plt.figure()
        plt.hist(pred_recorded_list, label="recorded", bins=40, color="c", alpha=0.5)
        plt.hist(pred_synth_list, label="synthesized", bins=40, color="m", alpha=0.5)
        plt.legend()
        plt.grid()
        plt.savefig(str(save_path_spk / "ranking_result.png"))
        plt.close()
        
        
if __name__ == "__main__":
    main()