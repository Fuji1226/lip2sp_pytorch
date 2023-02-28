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

from parallelwavegan.pwg_train_raw import make_model as make_pwg
from train_tts_raw import make_model as make_tts
from train_face_gen_raw import make_model as make_face_gen
from train_nar_raw import make_model as make_lip2sp
from utils import get_path_test_raw, make_test_loader_tts_face_raw, load_pretrained_model, make_test_loader_face_gen_raw, gen_data_separate, gen_data_concat
from data_check import save_data_tts, save_data_pwg, save_data, save_data_pwg, save_data_face_gen

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def check_attention_weight(att_w, cfg, filename, save_path):
    att_w = att_w.to('cpu').detach().numpy().copy()

    plt.figure()
    sns.heatmap(att_w, cmap="viridis", cbar=True)
    plt.title("attention weight")
    plt.xlabel("text")
    plt.ylabel("feature")

    plt.savefig(str(save_path / f"{filename}.png"))
    plt.close()


def generate(cfg, tts, face_gen, lip2sp, pwg, test_loader, dataset, train_loader, device, save_path):
    tts.eval()
    face_gen.eval()
    lip2sp.eval()
    pwg.eval()
    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)

    for batch in tqdm(test_loader, total=len(test_loader)):
        # face_genの入力フレームを学習データから取得
        batch_train = train_loader.next()
        wav, lip, feature, spk_emb, feature_len, lip_len, speaker, label = batch_train
        lip_first_frame = lip[..., 0]
        lip_first_frame = lip_first_frame.to(device)

        wav, text, feature, lip, stop_token, text_len, feature_len, lip_len, spk_emb, speaker, label = batch
        text = text.to(device)
        lip = lip.to(device)
        wav = wav.to(device)
        feature = feature.to(device)
        stop_token = stop_token.to(device)
        text_len = text_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)

        with torch.no_grad():
            # tts
            dec_output, feat_synth_tts, logit, att_w = tts(text, text_len, spk_emb=spk_emb)

            feature_len_synth = feat_synth_tts.shape[-1]
            lip_len_synth = torch.tensor(feat_synth_tts.shape[-1] // cfg.model.reduction_factor).unsqueeze(0)
            feature_synth_sep = gen_data_separate(
                feat_synth_tts,
                int(cfg.model.input_lip_sec * cfg.model.fps * cfg.model.reduction_factor),
                int(cfg.model.fps * cfg.model.reduction_factor),
            )

            lip_len_synth = lip_len_synth.expand(feature_synth_sep.shape[0])
            lip_first_frame = lip_first_frame.expand(feature_synth_sep.shape[0], -1, -1, -1)

            # face gen
            lip_synth = face_gen(lip_first_frame, feature_synth_sep, lip_len_synth)
            lip_synth = gen_data_concat(lip_synth, cfg.model.fps, lip_len_synth[0] % cfg.model.fps)

            lip_synth_sep = gen_data_separate(lip_synth, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
            spk_emb = spk_emb.expand(lip_synth_sep.shape[0], -1)

            # lip2sp
            feat_synth_lip2sp, _, _ = lip2sp(lip_synth_sep, lip_len_synth, spk_emb)
            feat_synth_lip2sp = gen_data_concat(
                feat_synth_lip2sp,
                int(cfg.model.fps * cfg.model.reduction_factor), 
                int((lip_len_synth[0] % cfg.model.fps) * cfg.model.reduction_factor),
            )

            # pwg
            noise = torch.randn(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length).to(device)
            wav_abs = pwg(noise, feature)

            noise = torch.randn(feat_synth_tts.shape[0], 1, feat_synth_tts.shape[-1] * cfg.model.hop_length).to(device)
            wav_tts = pwg(noise, feat_synth_tts)

            noise = torch.randn(feat_synth_lip2sp.shape[0], 1, feat_synth_lip2sp.shape[-1] * cfg.model.hop_length).to(device)
            wav_lip2sp = pwg(noise, feat_synth_lip2sp)

        data_save_path = save_path / speaker[0] / label[0]

        video_save_path = data_save_path / "video"
        video_save_path.mkdir(parents=True, exist_ok=True)

        tts_save_path = data_save_path / "tts"
        tts_save_path_gl = tts_save_path / "griffinlim"
        tts_save_path_pwg = tts_save_path / "pwg"
        tts_save_path_gl.mkdir(parents=True, exist_ok=True)
        tts_save_path_pwg.mkdir(parents=True, exist_ok=True)

        lip2sp_save_path = data_save_path / "lip2sp"
        lip2sp_save_path_gl= lip2sp_save_path / "griffinlim"
        lip2sp_save_path_pwg = lip2sp_save_path / "pwg"
        lip2sp_save_path_gl.mkdir(parents=True, exist_ok=True)
        lip2sp_save_path_pwg.mkdir(parents=True, exist_ok=True)

        # tts
        save_data_tts(
            cfg=cfg,
            save_path=tts_save_path_gl,
            wav=wav,
            feature=feature,
            output=feat_synth_tts,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        check_attention_weight(att_w[0], cfg, "attention", tts_save_path)
        save_data_pwg(
            cfg=cfg,
            save_path=tts_save_path_pwg,
            target=wav,
            output=wav_tts,
            ana_syn=wav_abs,
        )

        # lip2sp
        save_data(
            cfg=cfg,
            save_path=lip2sp_save_path_gl,
            wav=wav,
            lip=lip,
            feature=feature,
            output=feat_synth_lip2sp,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        save_data_pwg(
            cfg=cfg,
            save_path=lip2sp_save_path_pwg,
            target=wav,
            output=wav_lip2sp,
            ana_syn=wav_abs,
        )

        # face gen
        save_data_face_gen(
            cfg=cfg,
            save_path=video_save_path,
            wav=wav,
            target=lip,
            output=lip_synth,
            lip_mean=lip_mean,
            lip_std=lip_std,
        )


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # tts
    tts = make_tts(cfg, device)
    model_path_tts = Path(Path(f"~/lip2sp_pytorch/check_point/tts/face_aligned_0_50_gray/2023:01:08_10-33-05/mspec80_200.ckpt").expanduser())
    tts = load_pretrained_model(model_path_tts, tts, "model")

    # face gen
    face_gen, _, _, _, _ = make_face_gen(cfg, device)
    model_path_face_gen = Path(f"~/lip2sp_pytorch/check_point/face_gen/face/2023:02:24_16-27-37/mspec80_110.ckpt").expanduser()
    face_gen = load_pretrained_model(model_path_face_gen, face_gen, "gen")

    # lip2sp
    lip2sp = make_lip2sp(cfg, device)
    model_path_lip2sp = Path(f"~/lip2sp_pytorch/check_point/nar/face/2023:02:24_17-25-12/mspec80_340.ckpt").expanduser()
    lip2sp = load_pretrained_model(model_path_lip2sp, lip2sp, "model")

    # pwg
    pwg, disc = make_pwg(cfg, device)
    model_path_pwg = Path(f"~/lip2sp_pytorch/parallelwavegan/check_point/default/face_aligned_0_50_gray/2023:01:30_15-38-44/mspec80_300.ckpt").expanduser()
    pwg = load_pretrained_model(model_path_pwg, pwg, "gen")

    cfg.train.face_or_lip = model_path_lip2sp.parents[1].name
    cfg.test.face_or_lip = model_path_lip2sp.parents[1].name

    data_dir, bbox_dir, landmark_dir, df_list, save_path_list, train_df = get_path_test_raw(cfg, model_path_lip2sp)

    for df, save_path in zip(df_list, save_path_list):
        _, _, train_loader, train_dataset = make_test_loader_face_gen_raw(data_dir, bbox_dir, landmark_dir, train_df, df, cfg)
        train_loader = iter(train_loader)
        test_loader, test_dataset = make_test_loader_tts_face_raw(data_dir, bbox_dir, landmark_dir, train_df, df, cfg)
        
        generate(
            cfg=cfg,
            tts=tts,
            face_gen=face_gen,
            lip2sp=lip2sp,
            pwg=pwg,
            test_loader=test_loader,
            dataset=test_dataset,
            train_loader=train_loader,
            device=device,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()