from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow
import hydra
import pyworld

from data_process.feature import wav2mel, wav2world

color_or_gray = "color"

lip03_path = Path("~/lip2sp_pytorch/result/nar/generate/lip_cropped_0.3_50_gray/2022:12:09_13-29-45/mspec80_300/test_data/audio/F01_kablab").expanduser()
lip08_path = Path("~/lip2sp_pytorch/result/nar/generate/lip_cropped_0.8_50_gray/2022:12:09_13-46-31/mspec80_300/test_data/audio/F01_kablab").expanduser()
face_path = Path("~/lip2sp_pytorch/result/nar/generate/face_aligned_0_50_gray/2022:12:09_14-02-12/mspec80_300/test_data/audio/F01_kablab").expanduser()
face_delta_path = Path("~/lip2sp_pytorch/result/nar/generate/face_aligned_0_50_gray/2022:12:12_10-27-44/mspec80_300/test_data/audio/F01_kablab").expanduser()
face_time_masking_path = Path("~/lip2sp_pytorch/result/nar/generate/face_aligned_0_50_gray/2022:12:11_16-17-37/mspec80_300/test_data/audio/F01_kablab").expanduser()
ar_tf_path = Path("~/lip2sp_pytorch/result/default/generate/face_aligned_0_50_gray/2022:12:15_13-54-16/mspec80_360/test_data/audio").expanduser()
ar_tf_no_mask_path = Path("~/lip2sp_pytorch/result/default/generate/face_aligned_0_50_gray/2022:12:17_16-36-04/mspec80_360/test_data/audio").expanduser()
ar_ss_path = Path("~/lip2sp_pytorch/result/default/generate/face_aligned_0_50_gray/2022:12:15_14-11-36/mspec80_410_pre/test_data/audio").expanduser()

data_name = "ATR503_j01_0_mspec80"
lip03_path = lip03_path / data_name
lip08_path = lip08_path / data_name
face_path = face_path / data_name
face_delta_path = face_delta_path / data_name
face_time_masking_path = face_time_masking_path / data_name
ar_tf_path = ar_tf_path / data_name
ar_tf_no_mask_path = ar_tf_no_mask_path / data_name
ar_ss_path = ar_ss_path / data_name

if color_or_gray ==  "color":
    save_dir = Path("~/lip2sp_pytorch/thesis_result").expanduser()
elif color_or_gray == "gray":
    save_dir = Path("~/lip2sp_pytorch/thesis_result_gray").expanduser()
save_dir = save_dir / data_name
save_dir.mkdir(parents=True, exist_ok=True)


def plot_spec(data, cfg, title):
    if color_or_gray == "color":
        cmap = "viridis"
    elif color_or_gray == "gray":
        cmap == "gray"
    specshow(
        data=data,
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap=cmap,
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title(title)


def wav2f0(wav, cfg, f0_floor, f0_ceil):
    f0, _ = pyworld.harvest(
        wav, 
        cfg.model.sampling_rate,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=cfg.model.frame_period,
    )
    return f0


def plot_f0(f0_orig, f0_gen, cfg, title, y_lim_upper):
    time = np.arange(0, f0_orig.shape[0]) / 100
    if color_or_gray == "color":
        plt.plot(time, f0_orig, label="Ground Truth")
        plt.plot(time, f0_gen, label="Synthesis")
    elif color_or_gray == "gray":
        plt.plot(time, f0_orig, label="Ground Truth", color="black", linestyle="solid")
        plt.plot(time, f0_gen, label="Synthesis", color="gray", linestyle="dashed")
    plt.xlabel("Time[s]")
    plt.ylabel("f0[Hz]")
    plt.xlim(0, time[-1])
    plt.ylim(0, y_lim_upper)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.2)
    plt.grid()


def plot_clf0(f0_orig, f0_gen, cfg, title, y_lim_upper):
    time = np.arange(0, f0_orig.shape[0]) / 100
    plt.plot(time, f0_orig, label="Ground Truth")
    plt.plot(time, f0_gen, label="Synthesis")
    plt.xlabel("Time[s]")
    plt.ylabel("f0[Hz]")
    plt.xlim(0, time[-1])
    # plt.ylim(0, y_lim_upper)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.2)
    plt.grid()


def input_comparison(cfg):
    wav_orig, _ = librosa.load(str(lip03_path / "input.wav"), sr=cfg.model.sampling_rate)
    wav_lip03, _ = librosa.load(str(lip03_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_lip08, _ = librosa.load(str(lip08_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_face, _ = librosa.load(str(face_path / "generate.wav"), sr=cfg.model.sampling_rate)

    mel_orig = wav2mel(wav_orig, cfg, ref_max=True)
    mel_lip03 = wav2mel(wav_lip03, cfg, ref_max=True)
    mel_lip08 = wav2mel(wav_lip08, cfg, ref_max=True)
    mel_face = wav2mel(wav_face, cfg, ref_max=True)

    plt.figure(figsize=(8, 12))

    plt.subplot(4, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")

    plt.subplot(4, 1, 2)
    plot_spec(mel_lip03, cfg, "Lip Only")

    plt.subplot(4, 1, 3)
    plot_spec(mel_lip08, cfg, "Lip Wide")

    plt.subplot(4, 1, 4)
    plot_spec(mel_face, cfg, "Face")

    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_input_comparison.png"))
    plt.close()

    wav_orig = wav_orig.astype('float64')
    wav_lip03 = wav_lip03.astype('float64')
    wav_lip08 = wav_lip08.astype('float64')
    wav_face = wav_face.astype('float64')

    f0_floor = pyworld.default_f0_floor
    f0_ceil = pyworld.default_f0_ceil

    f0_orig = wav2f0(wav_orig, cfg, f0_floor, f0_ceil)
    f0_lip03 = wav2f0(wav_lip03, cfg, f0_floor, f0_ceil)
    f0_lip08 = wav2f0(wav_lip08, cfg, f0_floor, f0_ceil)
    f0_face = wav2f0(wav_face, cfg, f0_floor, f0_ceil)

    plt.figure(figsize=(8, 9))
    ylim_upper = 500

    plt.subplot(3, 1, 1)
    plot_f0(f0_orig, f0_lip03, cfg, "Lip Only", ylim_upper)

    plt.subplot(3, 1, 2)
    plot_f0(f0_orig, f0_lip08, cfg, "Lip Wide", ylim_upper)

    plt.subplot(3, 1, 3)
    plot_f0(f0_orig, f0_face, cfg, "Face", ylim_upper)

    plt.tight_layout()
    plt.savefig(str(save_dir / "f0_input_comparison.png"))
    plt.close()


    _, clf0_orig, _, _, _, _ = wav2world(wav_orig, cfg.model.sampling_rate, cfg)
    _, clf0_lip03, _, _, _, _ = wav2world(wav_lip03, cfg.model.sampling_rate, cfg)
    _, clf0_lip08, _, _, _, _ = wav2world(wav_lip08, cfg.model.sampling_rate, cfg)
    _, clf0_face, _, _, _, _ = wav2world(wav_face, cfg.model.sampling_rate, cfg)

    plt.figure(figsize=(8, 9))

    plt.subplot(3, 1, 1)
    plot_clf0(clf0_orig, clf0_lip03, cfg, "Lip Only", ylim_upper)

    plt.subplot(3, 1, 2)
    plot_clf0(clf0_orig, clf0_lip08, cfg, "Lip Wide", ylim_upper)

    plt.subplot(3, 1, 3)
    plot_clf0(clf0_orig, clf0_face, cfg, "Face", ylim_upper)

    plt.tight_layout()
    plt.savefig(str(save_dir / "clf0_input_comparison.png"))
    plt.close()


def delta_comparison(cfg):
    wav_orig, _ = librosa.load(str(face_path / "input.wav"), sr=cfg.model.sampling_rate)
    wav_face, _ = librosa.load(str(face_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_face_delta, _ = librosa.load(str(face_delta_path / "generate.wav"), sr=cfg.model.sampling_rate)

    mel_orig = wav2mel(wav_orig, cfg, ref_max=True)
    mel_face = wav2mel(wav_face, cfg, ref_max=True)
    mel_face_delta = wav2mel(wav_face_delta, cfg, ref_max=True)

    plt.figure(figsize=(8, 9))

    plt.subplot(3, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")

    plt.subplot(3, 1, 2)
    plot_spec(mel_face, cfg, "Face")

    plt.subplot(3, 1, 3)
    plot_spec(mel_face_delta, cfg, "Face + Delta")

    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_delta_comparison.png"))
    plt.close()


def time_masking_comparison(cfg):
    wav_orig, _ = librosa.load(str(face_path / "input.wav"), sr=cfg.model.sampling_rate)
    wav_face, _ = librosa.load(str(face_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_face_mask, _ = librosa.load(str(face_time_masking_path / "generate.wav"), sr=cfg.model.sampling_rate)

    mel_orig = wav2mel(wav_orig, cfg, ref_max=True)
    mel_face = wav2mel(wav_face, cfg, ref_max=True)
    mel_face_mask = wav2mel(wav_face_mask, cfg, ref_max=True)

    plt.figure(figsize=(8, 9))

    plt.subplot(3, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")

    plt.subplot(3, 1, 2)
    plot_spec(mel_face, cfg, "Face")

    plt.subplot(3, 1, 3)
    plot_spec(mel_face_mask, cfg, "Face + Time Masking")

    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_time_masking_comparison.png"))
    plt.close()

    wav_orig = wav_orig.astype('float64')
    wav_face = wav_face.astype('float64')
    wav_face_mask = wav_face_mask.astype('float64')

    f0_floor = pyworld.default_f0_floor
    f0_ceil = pyworld.default_f0_ceil

    f0_orig = wav2f0(wav_orig, cfg, f0_floor, f0_ceil)
    f0_face = wav2f0(wav_face, cfg, f0_floor, f0_ceil)
    f0_face_mask = wav2f0(wav_face_mask, cfg, f0_floor, f0_ceil)

    plt.figure(figsize=(8, 6))
    ylim_upper = 550

    plt.subplot(2, 1, 1)
    plot_f0(f0_orig, f0_face, cfg, "Face", ylim_upper)

    plt.subplot(2, 1, 2)
    plot_f0(f0_orig, f0_face_mask, cfg, "Face Time Masking", ylim_upper)

    plt.tight_layout()
    plt.savefig(str(save_dir / "f0_time_masking_comparison.png"))
    plt.close()



def ar_nar_comparison(cfg):
    wav_orig, _ = librosa.load(str(face_path / "input.wav"), sr=cfg.model.sampling_rate)
    wav_face_mask, _ = librosa.load(str(face_time_masking_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_ar_tf, _ = librosa.load(str(ar_tf_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_ar_ss, _ = librosa.load(str(ar_ss_path / "generate.wav"), sr=cfg.model.sampling_rate)

    mel_orig = wav2mel(wav_orig, cfg, ref_max=True)
    mel_face_mask = wav2mel(wav_face_mask, cfg, ref_max=True)
    mel_ar_tf = wav2mel(wav_ar_tf, cfg, ref_max=True)
    mel_ar_ss = wav2mel(wav_ar_ss, cfg, ref_max=True)

    plt.figure(figsize=(8, 12))

    plt.subplot(4, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")

    plt.subplot(4, 1, 2)
    plot_spec(mel_face_mask, cfg, "Non-autoregressive Model")

    plt.subplot(4, 1, 3)
    plot_spec(mel_ar_tf, cfg, "Autoregressive Model : Teacher Forcing")

    plt.subplot(4, 1, 4)
    plot_spec(mel_ar_ss, cfg, "Autoregressive Model : Scheduled Sampling")

    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_ar_nar_comparison.png"))
    plt.close()

    wav_orig = wav_orig.astype('float64')
    wav_face_mask = wav_face_mask.astype('float64')
    wav_ar_tf = wav_ar_tf.astype('float64')
    wav_ar_ss = wav_ar_ss.astype('float64')

    f0_floor = pyworld.default_f0_floor
    f0_ceil = pyworld.default_f0_ceil

    f0_orig = wav2f0(wav_orig, cfg, f0_floor, f0_ceil)
    f0_face_mask = wav2f0(wav_face_mask, cfg, f0_floor, f0_ceil)
    f0_ar_tf = wav2f0(wav_ar_tf, cfg, f0_floor, f0_ceil)
    f0_ar_ss = wav2f0(wav_ar_ss, cfg, f0_floor, f0_ceil)

    plt.figure(figsize=(8, 9))
    ylim_upper = 520

    plt.subplot(3, 1, 1)
    plot_f0(f0_orig, f0_face_mask, cfg, "Non-autoregressice Model", ylim_upper)

    plt.subplot(3, 1, 2)
    plot_f0(f0_orig, f0_ar_tf, cfg, "Autoregressive Model : Teacher Forcing", ylim_upper)

    plt.subplot(3, 1, 3)
    plot_f0(f0_orig, f0_ar_ss, cfg, "Autoregressive Model : Scheduled Sampling", ylim_upper)

    plt.tight_layout()
    plt.savefig(str(save_dir / "f0_ar_nar_comparison.png"))
    plt.close()


def tf_time_masking_comparison(cfg):
    wav_orig, _ = librosa.load(str(face_path / "input.wav"), sr=cfg.model.sampling_rate)
    wav_ar_tf, _ = librosa.load(str(ar_tf_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_ar_tf_no_mask, _ = librosa.load(str(ar_tf_no_mask_path / "generate.wav"), sr=cfg.model.sampling_rate)

    mel_orig = wav2mel(wav_orig, cfg, ref_max=True)
    mel_ar_tf = wav2mel(wav_ar_tf, cfg, ref_max=True)
    mel_ar_tf_no_mask = wav2mel(wav_ar_tf_no_mask, cfg, ref_max=True)

    plt.figure(figsize=(8, 9))

    plt.subplot(3, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")

    plt.subplot(3, 1, 2)
    plot_spec(mel_ar_tf, cfg, "When Time Masking is Used")

    plt.subplot(3, 1, 3)
    plot_spec(mel_ar_tf_no_mask, cfg, "When Time Masking is Not Used")

    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_tf_time_masking_comparison.png"))
    plt.close()


def ar_nar_comparison_presen(cfg):
    wav_orig, _ = librosa.load(str(face_path / "input.wav"), sr=cfg.model.sampling_rate)
    wav_face_mask, _ = librosa.load(str(face_time_masking_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_ar_tf, _ = librosa.load(str(ar_tf_path / "generate.wav"), sr=cfg.model.sampling_rate)
    wav_ar_ss, _ = librosa.load(str(ar_ss_path / "generate.wav"), sr=cfg.model.sampling_rate)

    mel_orig = wav2mel(wav_orig, cfg, ref_max=True)
    mel_face_mask = wav2mel(wav_face_mask, cfg, ref_max=True)
    mel_ar_tf = wav2mel(wav_ar_tf, cfg, ref_max=True)
    mel_ar_ss = wav2mel(wav_ar_ss, cfg, ref_max=True)

    plt.figure(figsize=(8, 9))
    plt.subplot(3, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")
    plt.subplot(3, 1, 2)
    plot_spec(mel_ar_tf, cfg, "Autoregressive Model : Teacher Forcing")
    plt.subplot(3, 1, 3)
    plot_spec(mel_ar_ss, cfg, "Autoregressive Model : Scheduled Sampling")
    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_ar_comparison_presen.png"))
    plt.close()

    plt.figure(figsize=(8, 9))
    plt.subplot(3, 1, 1)
    plot_spec(mel_orig, cfg, "Ground Truth")
    plt.subplot(3, 1, 2)
    plot_spec(mel_face_mask, cfg, "Non-Autoregressive Model")
    plt.subplot(3, 1, 3)
    plot_spec(mel_ar_ss, cfg, "Autoregressive Model : Scheduled Sampling")
    plt.tight_layout()
    plt.savefig(str(save_dir / "mel_ar_nar_comparison_presen.png"))
    plt.close()

    # wav_orig = wav_orig.astype('float64')
    # wav_face_mask = wav_face_mask.astype('float64')
    # wav_ar_tf = wav_ar_tf.astype('float64')
    # wav_ar_ss = wav_ar_ss.astype('float64')

    # f0_floor = pyworld.default_f0_floor
    # f0_ceil = pyworld.default_f0_ceil

    # f0_orig = wav2f0(wav_orig, cfg, f0_floor, f0_ceil)
    # f0_face_mask = wav2f0(wav_face_mask, cfg, f0_floor, f0_ceil)
    # f0_ar_tf = wav2f0(wav_ar_tf, cfg, f0_floor, f0_ceil)
    # f0_ar_ss = wav2f0(wav_ar_ss, cfg, f0_floor, f0_ceil)

    # plt.figure(figsize=(8, 9))
    # ylim_upper = 520
    # plt.subplot(3, 1, 1)
    # plot_f0(f0_orig, f0_face_mask, cfg, "Non-autoregressice Model", ylim_upper)
    # plt.subplot(3, 1, 2)
    # plot_f0(f0_orig, f0_ar_tf, cfg, "Autoregressive Model : Teacher Forcing", ylim_upper)
    # plt.subplot(3, 1, 3)
    # plot_f0(f0_orig, f0_ar_ss, cfg, "Autoregressive Model : Scheduled Sampling", ylim_upper)
    # plt.tight_layout()
    # plt.savefig(str(save_dir / "f0_ar_nar_comparison.png"))
    # plt.close()


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    input_comparison(cfg)
    delta_comparison(cfg)
    time_masking_comparison(cfg)
    ar_nar_comparison(cfg)
    tf_time_masking_comparison(cfg)
    ar_nar_comparison_presen(cfg)


if __name__ == "__main__":
    main()