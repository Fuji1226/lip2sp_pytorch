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
import pandas as pd
import pyopenjtalk
import functools
import itertools

from train_tts_vae import make_model as make_tts
from train_face_gen import make_model as make_face_gen
from parallelwavegan.pwg_train import make_model as make_pwg
from utils import load_pretrained_model, get_path_train, make_train_val_loader_tts, gen_data_separate, gen_data_concat
from data_process.phoneme_encode import classes2index_tts, pp_symbols
from dataset.utils import text_dir

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def preprocess(text, class_to_id):
    text = pyopenjtalk.extract_fullcontext(text)
    text = pp_symbols(text)
    text = [class_to_id[t] for t in text]
    return text
    

def get_text_jsut():
    text_dir = Path("~/dataset/jsut_ver1.1").expanduser()
    corpus_list = [
        "basic5000", "countersuffix26", "loanword128", "onomatopee300", "precedent130", 
        "travel1000", "utparaphrase512", "voiceactress100"
    ]
    df_list = []
    
    for corpus in corpus_list:
        corpus_text_path = text_dir / corpus / "transcript_utf8.txt"
        df = pd.read_csv(str(corpus_text_path), header=None)
        df_fix = df.copy()
        df_fix["filename"] = df[0].apply(lambda x : str(x.split(":")[0]))
        df_fix["text"] = df[0].apply(lambda x : str(x.split(":")[1]))
        df_fix = df_fix.drop(columns=[0])
        df_fix = df_fix.reset_index(drop=True)
        if corpus == "basic5000":
            df_fix = df_fix[2500:]
        df_list.append(df_fix)
        
    df = pd.concat(df_list)
    return df


def get_text_wiki():
    text_dir = Path(f"~/dataset/wiki/datafiles/sentences/limit_130/text_with_filename.csv").expanduser()
    df = pd.read_csv(str(text_dir), nrows=100000, usecols=["filename", "text"])
    return df


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tts = make_tts(cfg, device)
    tts_path = Path(f"~/lip2sp_pytorch/check_point/tts_vae/face_aligned_0_50_gray/2023:04:09_11-09-23/mspec80_200.ckpt").expanduser()
    tts = load_pretrained_model(tts_path, tts, "model")
    
    face_gen, _, _, _, _ = make_face_gen(cfg, device)
    face_gen_path = Path(f"~/lip2sp_pytorch/check_point/face_gen/face_aligned_0_50_gray/2023:04:10_13-15-28/mspec80_100.ckpt").expanduser()
    face_gen = load_pretrained_model(face_gen_path, face_gen, "gen")
    
    pwg, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_300.ckpt").expanduser()
    pwg = load_pretrained_model(model_path_pwg, pwg, "gen")
    
    cfg.train.face_or_lip = face_gen_path.parents[1].name
    cfg.test.face_or_lip = face_gen_path.parents[1].name
    cfg.train.batch_size = 1
    
    train_data_root, val_data_root, ckpt_path, _, ckpt_time = get_path_train(cfg, current_time)
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_tts(cfg, train_data_root, val_data_root)
    loader_for_vae = itertools.cycle(iter(train_loader))
    
    save_path = Path(f"~/dataset/lip/np_files_synth_corpus/{cfg.train.face_or_lip}/train").expanduser()
    
    df_jsut = get_text_jsut()
    df_wiki = get_text_wiki()
    df = pd.concat([df_jsut, df_wiki]).reset_index(drop=True)
    class_to_id, id_to_class = classes2index_tts()
    print(f"df = {df.shape}")
    
    lip_mean = train_dataset.lip_mean.to(device)
    lip_std = train_dataset.lip_std.to(device)
    feat_mean = train_dataset.feat_mean.to(device)
    feat_std = train_dataset.feat_std.to(device)
    lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    feat_mean = feat_mean.unsqueeze(1)
    feat_std = feat_std.unsqueeze(1)
    tts.eval()
    face_gen.eval()
    pwg.eval()
    
    for i in tqdm(range(df.shape[0])):
        batch_train = next(loader_for_vae)
        wav, lip, feature, _, stop_token, spk_emb, feature_len, lip_len, _, speaker, speaker_idx, label = batch_train
        feature_for_vae = feature.to(device)
        feature_for_vae_len = feature_len.to(device)
        lip_first_frame = lip[..., 0]
        lip_first_frame = lip_first_frame.to(device)
        
        data = df.iloc[i]
        filename = data["filename"]
        text = data["text"]
        with open(str(text_dir / f"{filename}.txt"), "w") as f:
            f.write(text)
        
        text = preprocess(text, class_to_id)
        text = torch.tensor([text]).to(device)
        text_len = torch.tensor([text.shape[1]]).to(device)

        with torch.no_grad():
            dec_output, output_feature, logit, att_w, mu, logvar = tts(text, text_len, feature_for_vae, feature_for_vae_len, feature_target=None, spk_emb=spk_emb)
            noise = torch.randn(output_feature.shape[0], 1, output_feature.shape[-1] * cfg.model.hop_length).to(device=device, dtype=output_feature.dtype)
            output_wav = pwg(noise, output_feature)
            
        output_feature_sep = gen_data_separate(
            output_feature,
            int(cfg.model.input_lip_sec * cfg.model.sampling_rate // cfg.model.hop_length), 
            cfg.model.sampling_rate // cfg.model.hop_length,
        )
        output_feature_len = torch.tensor([output_feature.shape[-1]])
        output_lip_len = output_feature_len // ((cfg.model.sampling_rate // cfg.model.hop_length) // cfg.model.fps)
        output_lip_len = output_lip_len.expand(output_feature_sep.shape[0])
        lip_first_frame = lip_first_frame.expand(output_feature_sep.shape[0], -1, -1, -1)
        
        with torch.no_grad():
            output_lip = face_gen(lip_first_frame, output_feature_sep, output_lip_len)
            
        output_lip = gen_data_concat(output_lip, cfg.model.fps, output_lip_len[0] % cfg.model.fps)
        
        output_wav = output_wav[0]
        output_lip = output_lip[0]
        output_feature = output_feature[0]
        
        output_lip = torch.mul(output_lip, lip_std)
        output_lip = torch.add(output_lip, lip_mean)
        output_feature *= feat_std
        output_feature += feat_mean
        
        output_lip = output_lip.cpu().numpy()   # (C, H, W, T)
        output_feature = output_feature.cpu().numpy().T     # (T, C)
        output_wav = output_wav.cpu().numpy()
        output_wav /= np.max(np.abs(output_wav))
        
        _save_path = save_path / speaker[0] / cfg.model.name
        _save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(_save_path / filename),
            wav=output_wav,
            lip=output_lip,
            feature=output_feature,
        )


if __name__ == "__main__":
    main()