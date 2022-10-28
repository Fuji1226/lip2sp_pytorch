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


def calc_accuracy(data_dir, save_path, cfg, filename=None, process_times=None):
    wb_pesq = PerceptualEvaluationSpeechQuality(cfg.model.sampling_rate, 'wb')
    stoi = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, False)

    pesq_list = []
    stoi_list = []
    duration = []
    rmse_power_list = []
    rmse_f0_list_librosa = []
    vuv_acc_list_librosa = []
    rmse_f0_list_world = []
    vuv_acc_list_world = []
    mcd_list = []
    iter_cnt = 0

    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                if "generate" in Path(file).stem:
                    iter_cnt += 1
                    print(f"iter_cnt : {iter_cnt}")
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
                    print(f"PESQ = {p}")
                    print(f"STOI = {s}")

                    wav_gen = wav_gen.to("cpu").numpy()
                    wav_in = wav_in.to("cpu").numpy()

                    # powerのrmse
                    power_gen = librosa.feature.rms(y=wav_gen, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
                    power_gen = fill_nan(power_gen)
                    power_gen = librosa.amplitude_to_db(power_gen, ref=np.max)
                    power_in = librosa.feature.rms(y=wav_in, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
                    power_in = fill_nan(power_in)
                    power_in = librosa.amplitude_to_db(power_in, ref=np.max)
                    rmse_power = np.sqrt(np.mean((power_gen - power_in)**2))
                    rmse_power_list.append(rmse_power)
                    print(f"rmse_power = {rmse_power}")

                    # rmse f0 & vuv accuracy by librosa
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

                    rmse_f0 = (f0_gen - f0_in) ** 2
                    rmse_f0 = np.where(vuv_in == 1, rmse_f0, 0)     # 有声区間のみ
                    rmse_f0 = np.sqrt(np.mean(rmse_f0))

                    vuv_acc = np.sum((vuv_gen == vuv_in)) / vuv_in.size
                    vuv_acc *= 100
                    rmse_f0_list_librosa.append(rmse_f0)
                    vuv_acc_list_librosa.append(vuv_acc)
                    print(f"rmse_f0_librosa = {rmse_f0}, vuv_accuracy_librosa = {vuv_acc}")

                    # mel cepstral distortion
                    wav_gen = wav_gen.astype(np.float64)
                    f0_gen, timeaxis_gen = pyworld.harvest(wav_gen, fs, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
                    sp_gen = pyworld.cheaptrick(wav_gen, f0_gen, timeaxis_gen, fs)    
                    wav_in = wav_in.astype(np.float64)
                    f0_in, timeaxis_in = pyworld.harvest(wav_in, fs, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
                    sp_in = pyworld.cheaptrick(wav_in, f0_in, timeaxis_in, fs)
                    alpha = pysptk.util.mcepalpha(fs)
                    mcep_gen = pysptk.mcep(sp_gen, order=cfg.model.mcep_order - 1, alpha=alpha, itype=4)[:, 1:]
                    mcep_in = pysptk.mcep(sp_in, order=cfg.model.mcep_order - 1, alpha=alpha, itype=4)[:, 1:]
                    mcd = melcd(mcep_gen, mcep_in, lengths=None)
                    mcd_list.append(mcd)
                    print(f"mcd = {mcd}")

                    # rmse f0 & vuv accuracy by world
                    ap_gen = pyworld.d4c(wav_gen, f0_gen, timeaxis_gen, fs, threshold=0.85)
                    vuv_flag_gen = (ap_gen[:, 0] < 0.5) * (f0_gen > 1.0)
                    vuv_gen = vuv_flag_gen.astype('int')
                    ap_in = pyworld.d4c(wav_in, f0_in, timeaxis_in, fs, threshold=0.85)
                    vuv_flag_in = (ap_in[:, 0] < 0.5) * (f0_gen > 1.0)
                    vuv_in = vuv_flag_in.astype('int')

                    rmse_f0 = (f0_gen - f0_in) ** 2
                    rmse_f0 = np.where(vuv_in == 1, rmse_f0, 0)     # 有声区間のみ
                    rmse_f0 = np.sqrt(np.mean(rmse_f0))

                    vuv_acc = np.sum((vuv_gen == vuv_in)) / vuv_in.size
                    vuv_acc *= 100
                    rmse_f0_list_world.append(rmse_f0)
                    vuv_acc_list_world.append(vuv_acc)
                    print(f"rmse_f0_world = {rmse_f0}, vuv_accuracy_world = {vuv_acc}")
                    print("")

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
    rmse_f0_librosa = sum(rmse_f0_list_librosa) / len(rmse_f0_list_librosa)
    vuv_acc_librosa = sum(vuv_acc_list_librosa) / len(vuv_acc_list_librosa)
    rmse_f0_world = sum(rmse_f0_list_world) / len(rmse_f0_list_world)
    vuv_acc_world = sum(vuv_acc_list_world) / len(vuv_acc_list_world)
    mcd = sum(mcd_list) / len(mcd_list)

    file_name = save_path / f"accuracy.txt"
    with open(str(file_name), "a") as f:
        f.write("--- Objective Evaluation Metrics ---\n")
        f.write(f"PESQ = {pesq:f}\n")
        f.write(f"STOI = {stoi:f}\n")
        f.write(f"rmse power = {rmse_power:f}dB\n")
        f.write(f"rmsef0 librosa = {rmse_f0_librosa:f}\n")
        f.write(f"vuv accuracy librosa = {vuv_acc_librosa:f}%\n")
        f.write(f"rmse f0 world = {rmse_f0_world:f}\n")
        f.write(f"vuv accuracy world = {vuv_acc_world:f}%\n")
        f.write(f"mel cepstral distortion = {mcd:f}dB\n")

        if process_times is not None:
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
    iter_cnt = 0

    for p_s, p_m in zip(data_path_same, data_path_mix):
        iter_cnt += 1
        print(f"iter_cnt : {iter_cnt}")
        wav_same, fs = librosa.load(str(p_s), sr=cfg.model.sampling_rate, mono=True)
        wav_mix, fs = librosa.load(str(p_m), sr=cfg.model.sampling_rate, mono=True)

        _, sp_same, _ = pyworld.wav2world(wav_same.astype(np.double), fs=cfg.model.sampling_rate, frame_period=cfg.model.frame_period, fft_size=fft_size)
        mgc_same = pysptk.sptk.mcep(sp_same, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)    # (T, C)
        _, sp_mix, _ = pyworld.wav2world(wav_mix.astype(np.double), fs=cfg.model.sampling_rate,frame_period=cfg.model.frame_period, fft_size=fft_size)
        mgc_mix = pysptk.sptk.mcep(sp_mix, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

        ref_frame_no = len(mgc_same)
        min_cost, wp = librosa.sequence.dtw(mgc_same[:, 1:].T, mgc_mix[:, 1:].T)
        mcd = melcd(mgc_same[wp[:,0]], mgc_mix[wp[:,1]] , lengths=None)
        mcd_list.append(mcd)
        print(f"mcd = {mcd}")

    mcd = sum(mcd_list) / len(mcd_list)

    save_path = data_root_same.parents[2] / "accuracy.txt"
    with open(str(save_path), "a") as f:
        f.write(f"{data_root_same.parents[1].name}\n")
        f.write(f"mcd = {mcd:f}dB\n")
