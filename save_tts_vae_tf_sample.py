import hydra
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import seaborn as sns

from train_tts_vae import make_model
from utils import load_pretrained_model, get_path_train, make_train_val_loader_tts

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def save_tf_sample(cfg, model, train_loader, train_dataset, device, save_path):
    model.eval()
    lip_mean = train_dataset.lip_mean.to(device)
    lip_std = train_dataset.lip_std.to(device)
    feat_mean = train_dataset.feat_mean.to(device)
    feat_std = train_dataset.feat_std.to(device)
    lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    feat_mean = feat_mean.unsqueeze(1)
    feat_std = feat_std.unsqueeze(1)
    
    for batch in tqdm(train_loader):
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        lip = lip.to(device)
        text = text.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)

        with torch.no_grad():
            dec_output, output, logit, att_w, mu, logvar = model(text, text_len, feature, feature_len, feature_target=feature, spk_emb=spk_emb)
            
        assert feature.shape[-1] == output.shape[-1]
            
        _save_path = save_path / speaker[0] / cfg.model.name
        _save_path.mkdir(parents=True, exist_ok=True)
        
        wav = wav[0]
        lip = lip[0]
        output = output[0]
        lip = torch.mul(lip, lip_std)
        lip = torch.add(lip, lip_mean)
        output *= feat_std
        output += feat_mean
        
        wav = wav.cpu().numpy()
        lip = lip.cpu().numpy()
        output = output.cpu().numpy().T
        
        np.savez(
            str(_save_path / label[0]),
            wav=wav,
            lip=lip,
            feature=output,
        )
        

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    num_gen_epoch = 200
    model_path = Path(f"~/lip2sp_pytorch/check_point/tts_vae/face_aligned_0_50_gray/2023:04:09_11-09-23/mspec80_{num_gen_epoch}.ckpt").expanduser()     # F01
    model = make_model(cfg, device)
    model = load_pretrained_model(model_path, model, "model")
    cfg.train.face_or_lip = model_path.parents[1].name
    cfg.test.face_or_lip = model_path.parents[1].name
    cfg.train.batch_size = 1
    
    train_data_root, val_data_root, _, _, _ = get_path_train(cfg, current_time) 
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_tts(cfg, train_data_root, val_data_root)
    save_path = Path(f"~/dataset/lip/np_files_tts_tf_sample/{cfg.train.face_or_lip}/train").expanduser()
    
    save_tf_sample(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        save_path=save_path,
    )    


if __name__ == "__main__":
    main()