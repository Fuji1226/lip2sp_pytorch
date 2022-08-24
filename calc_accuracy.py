"""
PESQ,STOIなど客観評価指標の算定

使用方法
1. data_dirの変更
    PESQ, STOIを計算したい合成結果までのパスに変更

2. save_fileの変更
    とりあえずcsvの形式で保存するようにしています
    別々のファイルで保存しておきたい場合などに変更してください

3. 実行
    generateディレクトリに保存されると思います
"""
import os
from pathlib import Path

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import librosa
from data_process.transform_no_chainer import fill_nan



def calc_accuracy(data_dir, save_path, cfg, process_times):
    wb_pesq = PerceptualEvaluationSpeechQuality(cfg.model.sampling_rate, 'wb')
    stoi = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, False)

    pesq_list = []
    stoi_list = []
    duration = []
    rmse_power_list = []
    rmse_f0_list = []
    vuv_acc_list = []

    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                if "generate" in Path(file).stem:
                    wav_gen, _ = torchaudio.load(os.path.join(curdir, file))
                    wav_in, _ = torchaudio.load(os.path.join(curdir, "input.wav"))

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

                    power_gen = librosa.feature.rms(y=wav_gen, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
                    power_gen = fill_nan(power_gen)
                    power_gen = librosa.amplitude_to_db(power_gen, ref=np.max)
                    power_in = librosa.feature.rms(y=wav_in, frame_length=cfg.model.hop_length*2, hop_length=cfg.model.hop_length).squeeze()
                    power_in = fill_nan(power_in)
                    power_in = librosa.amplitude_to_db(power_in, ref=np.max)

                    # powerのrmse
                    rmse_power = np.sqrt(np.mean((power_gen - power_in)**2))
                    rmse_power_list.append(rmse_power)

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

                    # f0のrmse
                    rmse_f0 = np.sqrt(np.mean((f0_gen - f0_in)**2))
                    rmse_f0_list.append(rmse_f0)

                    # vuvの一致率
                    vuv_acc = np.sum((vuv_gen == vuv_in)) / vuv_in.size
                    vuv_acc *= 100
                    vuv_acc_list.append(vuv_acc)

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

    file_name = save_path / "accuracy.txt"
    with open(str(file_name), "a") as f:
        f.write("--- Objective Evaluation Metrics ---\n")
        f.write(f"PESQ = {pesq:f}\n")
        f.write(f"STOI = {stoi:f}\n")
        f.write(f"rmse_power = {rmse_power:f}dB\n")
        f.write(f"rmse_f0 = {rmse_f0:f}\n")
        f.write(f"vuv_accuracy = {vuv_acc:f}%\n")

        f.write("\n--- Duration and Process Time ---\n")
        f.write(f"duration_mean = {sum(duration) / len(duration):f}, process_time_mean = {sum(process_times) / len(process_times):f}\n")
        for dur, time in zip(duration, process_times):
            f.write(f"duration = {dur:f}, process_time = {time:f}\n")