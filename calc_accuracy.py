import os
from pathlib import Path
import hydra

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import librosa
from data_process.transform_no_chainer import fill_nan
from nnmnkwii.metrics import melcd
import pyworld
import pysptk


def calc_accuracy(data_dir, save_path, cfg, process_times):
    wb_pesq = PerceptualEvaluationSpeechQuality(cfg.model.sampling_rate, 'wb')
    stoi = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, False)

    pesq_list = []
    stoi_list = []
    duration = []
    rmse_power_list = []
    rmse_f0_list = []
    vuv_acc_list = []
    mcd_list = []

    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                if "generate" in Path(file).stem:
                    wav_gen, fs = torchaudio.load(os.path.join(curdir, file))
                    wav_in, fs = torchaudio.load(os.path.join(curdir, "input.wav"))

                    wav_gen = wav_gen.squeeze(0)
                    wav_in = wav_in.squeeze(0)

                    shorter_n_frame = int(min(wav_gen.shape[0], wav_in.shape[0]))
                    wav_gen = wav_gen[:shorter_n_frame]
                    wav_in = wav_in[:shorter_n_frame]
                    assert wav_gen.shape[0] == wav_in.shape[0]

                    # pesq, stoi
                    p = wb_pesq(wav_gen, wav_in)
                    s = stoi(wav_gen, wav_in)
                    pesq_list.append(p)
                    stoi_list.append(s)
                    duration.append(shorter_n_frame / cfg.model.sampling_rate)

                    wav_gen = wav_gen.to("cpu").detach().numpy().copy()
                    wav_in = wav_in.to("cpu").detach().numpy().copy()

                    # powerのrmse
                    power_gen = librosa.feature.rms(y=wav_gen, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
                    power_gen = fill_nan(power_gen)
                    power_gen = librosa.amplitude_to_db(power_gen, ref=np.max)
                    power_in = librosa.feature.rms(y=wav_in, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
                    power_in = fill_nan(power_in)
                    power_in = librosa.amplitude_to_db(power_in, ref=np.max)
                    rmse_power = np.sqrt(np.mean((power_gen - power_in)**2))
                    rmse_power_list.append(rmse_power)

                    # f0のrmse
                    f0_gen, vuv_gen, voiced_probs_gen = librosa.pyin(
                        y=wav_gen,
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=cfg.model.sampling_rate,
                        frame_length=cfg.model.win_length,
                        win_length=cfg.model.win_length // 2,
                        hop_length=cfg.model.hop_length,
                        fill_na=None,
                    )
                    f0_in, vuv_in, voiced_probs_in = librosa.pyin(
                        y=wav_in,
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=cfg.model.sampling_rate,
                        frame_length=cfg.model.win_length,
                        win_length=cfg.model.win_length // 2,
                        hop_length=cfg.model.hop_length,
                        fill_na=None,
                    )
                    rmse_f0 = np.sqrt(np.mean((f0_gen - f0_in)**2))
                    rmse_f0_list.append(rmse_f0)

                    # vuvの一致率
                    vuv_acc = np.sum((vuv_gen == vuv_in)) / vuv_in.size
                    vuv_acc *= 100
                    vuv_acc_list.append(vuv_acc)

                    # mel cepstral distortion
                    wav_gen = wav_gen.astype(np.float64)
                    f0_1, timeaxis_1 = pyworld.harvest(wav_gen, fs, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
                    sp1 = pyworld.cheaptrick(wav_gen, f0_1, timeaxis_1, fs)    
                    wav_in = wav_in.astype(np.float64)
                    f0_2, timeaxis_2 = pyworld.harvest(wav_in, fs, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
                    sp2 = pyworld.cheaptrick(wav_in, f0_2, timeaxis_2, fs)
                    coded_sp_1 = pyworld.code_spectral_envelope(sp1, fs, 24)
                    coded_sp_2 = pyworld.code_spectral_envelope(sp2, fs, 24)
                    mcd = melcd(coded_sp_1, coded_sp_2, lengths=None)
                    mcd_list.append(mcd)

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.scatter(duration, pesq_list)
    plt.xlabel("duration[s]")
    plt.ylabel("PESQ")
    plt.title("relationships between duration and PESQ")
    plt.grid()

    ax = plt.subplot(2, 1, 2)
    ax.scatter(duration, stoi_list)
    plt.xlabel("duration[s]")
    plt.ylabel("STOI")
    plt.title("relationships between duration and STOI")
    plt.grid()

    img_save_path = save_path / "check_PESQ_STOI.png"
    plt.tight_layout()
    plt.savefig(img_save_path)

    pesq = sum(pesq_list) / len(pesq_list)
    stoi = sum(stoi_list) / len(stoi_list)
    rmse_power = sum(rmse_power_list) / len(rmse_power_list)
    rmse_f0 = sum(rmse_f0_list) / len(rmse_f0_list)
    vuv_acc = sum(vuv_acc_list) / len(vuv_acc_list)
    mcd = sum(mcd_list) / len(mcd_list)

    file_name = save_path / "accuracy.txt"
    with open(str(file_name), "a") as f:
        f.write("--- Objective Evaluation Metrics ---\n")
        f.write(f"PESQ = {pesq:f}\n")
        f.write(f"STOI = {stoi:f}\n")
        f.write(f"rmse_power = {rmse_power:f}dB\n")
        f.write(f"rmse_f0 = {rmse_f0:f}\n")
        f.write(f"vuv_accuracy = {vuv_acc:f}%\n")
        f.write(f"mel cepstral distortion = {mcd:f}dB\n")

        f.write("\n--- Duration and Process Time ---\n")
        f.write(f"duration_mean = {sum(duration) / len(duration):f}, process_time_mean = {sum(process_times) / len(process_times):f}\n")
        for dur, time in zip(duration, process_times):
            f.write(f"duration = {dur:f}, process_time = {time:f}\n")


def calc_accuracy_vc(cfg, data_root_same, data_root_mix):
    data_path_same = list(sorted(data_root_same.glob("*/*generate.wav")))
    data_path_mix = list(sorted(data_root_mix.glob("*/*generate.wav")))

    fft_size = 512
    mcep_size = 34
    alpha = 0.65

    mcd_list = []

    for p_s, p_m in zip(data_path_same, data_path_mix):
        wav_same, fs = librosa.load(str(p_s), sr=cfg.model.sampling_rate, mono=True)
        wav_mix, fs = librosa.load(str(p_m), sr=cfg.model.sampling_rate, mono=True)

        _, sp_same, _ = pyworld.wav2world(wav_same.astype(np.double), fs=cfg.model.sampling_rate, frame_period=cfg.model.frame_period, fft_size=fft_size)
        mgc_same = pysptk.sptk.mcep(sp_same, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)    # (T, C)
        _, sp_mix, _ = pyworld.wav2world(wav_mix.astype(np.double), fs=cfg.model.sampling_rate,frame_period=cfg.model.frame_period, fft_size=fft_size)
        mgc_mix = pysptk.sptk.mcep(sp_mix, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

        ref_frame_no = len(mgc_same)
        min_cost, wp = librosa.sequence.dtw(mgc_same[:, 1:].T, mgc_mix[:, 1:].T)
        breakpoint()
        mcd = melcd(mgc_same[wp[:,0]], mgc_mix[wp[:,1]] , lengths=None)
        mcd_list.append(mcd)

    mcd = sum(mcd_list) / len(mcd_list)

    save_path = data_root_same.parents[2] / "accuracy.txt"
    with open(str(save_path), "a") as f:
        f.write(f"{data_root_same.parents[1].name}\n")
        f.write(f"mcd = {mcd:f}dB\n")
