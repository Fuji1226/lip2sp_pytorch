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
from librosa.display import specshow
import matplotlib.pyplot as plt

from data_check import save_data, save_data_pwg
from train_nar import make_model as make_lip2sp
from train_anomaly import make_model as make_anomaly
from utils import make_test_loader, get_path_test, load_pretrained_model, gen_data_separate, gen_data_concat
from calc_accuracy import calc_accuracy

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def show_mel(mel, fig, ax, cfg):
    img = specshow(
        data=mel, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
        ax=ax,
    )
    fig.colorbar(img, ax=ax)
    ax.set_xlabel("Time[s]")
    ax.set_ylabel("Frequency[Hz]")


def save(target_orig, target_recon, output_orig, output_recon, cfg, save_path):
    """
    Args:
        target_orig (_type_): (C, T)
        target_recon (_type_): 
        output_orig (_type_): 
        output_recon (_type_): 
        cfg (_type_): 
    """
    target_orig = target_orig.to("cpu").numpy()
    target_recon = target_recon.to("cpu").numpy()
    output_orig = output_orig.to("cpu").numpy()
    output_recon = output_recon.to("cpu").numpy()
    target_res = np.sqrt((target_orig - target_recon) ** 2)
    output_res = np.sqrt((output_orig - output_recon) ** 2)
    print(f"target mse = {np.mean(target_res)}")
    print(f"output mse = {np.mean(output_res)}")
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 9))
    show_mel(target_orig, fig, ax[0][0], cfg)
    ax[0][0].set_title("target_orig")
    show_mel(target_recon, fig, ax[1][0], cfg)
    ax[1][0].set_title("target_recon")
    show_mel(target_res, fig, ax[2][0], cfg)
    ax[2][0].set_title("target_res")
    show_mel(output_orig, fig, ax[0][1], cfg)
    ax[0][1].set_title("output_orig")
    show_mel(output_recon, fig, ax[1][1], cfg)
    ax[1][1].set_title("output_recon")
    show_mel(output_res, fig, ax[2][1], cfg)
    ax[2][1].set_title("output_res")
    
    plt.tight_layout()
    plt.savefig(str(save_path / "result.png"))
    plt.close()


def detect(cfg, lip2sp, detector, test_loader, dataset, device, save_path):
    lip2sp.eval()
    detector.eval()
    
    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        
        lip_sep = gen_data_separate(lip, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        lip_len = lip_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output, classifier_out, fmaps = lip2sp(lip_sep, lip_len, spk_emb)

        output = gen_data_concat(
            output, 
            int(cfg.model.fps * cfg.model.reduction_factor), 
            int((lip_len[0] % cfg.model.fps) * cfg.model.reduction_factor)
        )
        
        with torch.no_grad():
            feature_recon = detector(feature, torch.tensor([feature.shape[-1]]))
            output_recon = detector(output, torch.tensor([output.shape[-1]]))
        
        _save_path = save_path / speaker[0] / label[0]
        _save_path.mkdir(parents=True, exist_ok=True)
        save(
            target_orig=feature[0],
            target_recon=feature_recon[0],
            output_orig=output[0],
            output_recon=output_recon[0],
            cfg=cfg,
            save_path=_save_path,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    lip2sp = make_lip2sp(cfg, device)
    lip2sp_path = Path(f"~/lip2sp_pytorch/check_point/nar/face_aligned_0_50_gray/2023:04:04_17-15-09/mspec80_400.ckpt").expanduser()   # F01 face time masking
    lip2sp = load_pretrained_model(lip2sp_path, lip2sp, "model")
    cfg.train.face_or_lip = lip2sp_path.parents[1].name
    cfg.test.face_or_lip = lip2sp_path.parents[1].name
    
    detector = make_anomaly(cfg, device)
    anomaly_path = Path("~/lip2sp_pytorch/check_point/anomaly/face_aligned_0_50_gray/2023:04:08_08-33-42/mspec80_80.ckpt").expanduser()
    detector = load_pretrained_model(anomaly_path, detector, "model")
    
    data_root_list, save_path_list, train_data_root = get_path_test(cfg, anomaly_path)
    
    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)
        detect(
            cfg=cfg,
            lip2sp=lip2sp,
            detector=detector,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )
        
        
if __name__ == "__main__":
    main()