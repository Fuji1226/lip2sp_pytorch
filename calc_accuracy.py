import os
from pathlib import Path
import hydra

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import librosa
from data_process.transform import fill_nan
from nnmnkwii.metrics import melcd
import pyworld
import pysptk
import speech_recognition as sr
from subprocess import run
from jiwer import wer, cer
import MeCab
import pyopenjtalk
import pandas as pd
from collections import defaultdict
import re
import whisper
import torch


debug = False
abs_or_gen = "generate"


def wav2flac(data_dir):
    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav") and "generate" in Path(file).stem:
                file_gen = Path(curdir, file)
                file_in = Path(curdir, "input.wav")
                file_abs = Path(curdir, "abs.wav")
                file_gen_flac = Path(curdir, "generate.flac")
                file_in_flac = Path(curdir, "input.flac")
                file_abs_flac = Path(curdir, "abs.flac")

                cmd_gen = ["ffmpeg", "-y", "-i", f"{str(file_gen)}", "-vn", "-ar", "16000", "-ac", "1", "-acodec", "flac", "-f", "flac", f"{str(file_gen_flac)}"]
                cmd_in = ["ffmpeg", "-y", "-i", f"{str(file_in)}", "-vn", "-ar", "16000", "-ac", "1", "-acodec", "flac", "-f", "flac", f"{str(file_in_flac)}"]
                cmd_abs = ["ffmpeg", "-y", "-i", f"{str(file_abs)}", "-vn", "-ar", "16000", "-ac", "1", "-acodec", "flac", "-f", "flac", f"{str(file_abs_flac)}"]
                run(cmd_gen)
                run(cmd_in)
                run(cmd_abs)


def load_utt():
    csv_path = Path("~/lip2sp_pytorch/csv/ATR503.csv").expanduser()
    df = pd.read_csv(str(csv_path))
    df = df.values[-53:]
    return df


def calc_error_rate(utt, utt_pred):
    try:
        wer_gt = np.clip(wer(utt, utt_pred), a_min=0, a_max=1)
    except:
        wer_gt = 1.0
    return wer_gt


def calc_accuracy_new(data_dir, save_path, cfg, filename):
    speaker = data_dir.stem
    df = load_utt()
    wb_pesq_evaluator = PerceptualEvaluationSpeechQuality(cfg.model.sampling_rate, 'wb')
    stoi_evaluator = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, extended=False)
    estoi_evaluator = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, extended=True)
    speech_recognizer = whisper.load_model('large')
    mecab = MeCab.Tagger('-Owakati')

    pesq_abs_list = []
    pesq_generate_list = []
    stoi_abs_list = []
    stoi_generate_list = []
    estoi_abs_list = []
    estoi_generate_list = []
    wer_gt_list = []
    wer_abs_list = []
    wer_generate_list = []
    per_gt_list = []
    per_abs_list = []
    per_generate_list = []

    gt_data_path_list = list(data_dir.glob('**/gt.wav'))
    for i, gt_data_path in enumerate(gt_data_path_list):
        abs_data_path = Path(str(gt_data_path).replace('gt', 'abs'))
        generate_data_path = Path(str(gt_data_path).replace('gt', 'generate'))
        wav_gt, _ = librosa.load(str(gt_data_path), sr=cfg.model.sampling_rate)
        wav_abs, _ = librosa.load(str(abs_data_path), sr=cfg.model.sampling_rate)
        wav_generate, _ = librosa.load(str(generate_data_path), sr=cfg.model.sampling_rate)
        min_sample = min(wav_gt.shape[0], wav_abs.shape[0], wav_generate.shape[0])
        wav_gt = wav_gt[:min_sample]
        wav_abs = wav_abs[:min_sample]
        wav_generate = wav_generate[:min_sample]
        wav_gt = torch.from_numpy(wav_gt)
        wav_abs = torch.from_numpy(wav_abs)
        wav_generate = torch.from_numpy(wav_generate)

        for j in range(53):
            utt_num = df[j][2]
            if utt_num in gt_data_path.parents[0].name:
                utt = df[j][3]
                utt = utt.replace("。", "").replace("、", "")
                break

        pesq_abs = wb_pesq_evaluator(wav_abs, wav_gt)
        pesq_generate = wb_pesq_evaluator(wav_generate, wav_gt)
        stoi_abs = stoi_evaluator(wav_abs, wav_gt)
        stoi_generate = stoi_evaluator(wav_generate, wav_gt)
        estoi_abs = estoi_evaluator(wav_abs, wav_gt)
        estoi_generate = estoi_evaluator(wav_generate, wav_gt)
        pesq_abs_list.append(pesq_abs)
        pesq_generate_list.append(pesq_generate)
        stoi_abs_list.append(stoi_abs)
        stoi_generate_list.append(stoi_generate)
        estoi_abs_list.append(estoi_abs)
        estoi_generate_list.append(estoi_generate)

        utt_pred_gt = speech_recognizer.transcribe(str(gt_data_path), language='ja')['text'].replace('。', '').replace('、', '')
        utt_pred_abs = speech_recognizer.transcribe(str(abs_data_path), language='ja')['text'].replace('。', '').replace('、', '')
        utt_pred_generate = speech_recognizer.transcribe(str(generate_data_path), language='ja')['text'].replace('。', '').replace('、', '')
        utt_parse = mecab.parse(utt)
        utt_pred_gt_parse = mecab.parse(utt_pred_gt)
        utt_pred_abs_parse = mecab.parse(utt_pred_abs)
        utt_pred_generate_parse = mecab.parse(utt_pred_generate)
        wer_gt = calc_error_rate(utt_parse, utt_pred_gt_parse)
        wer_abs = calc_error_rate(utt_parse, utt_pred_abs_parse)
        wer_generate = calc_error_rate(utt_parse, utt_pred_generate_parse)

        utt_p = pyopenjtalk.g2p(utt)
        utt_pred_gt_p = pyopenjtalk.g2p(utt_pred_gt)
        utt_pred_abs_p = pyopenjtalk.g2p(utt_pred_abs)
        utt_pred_generate_p = pyopenjtalk.g2p(utt_pred_generate)
        per_gt = calc_error_rate(utt_p, utt_pred_gt_p)
        per_abs = calc_error_rate(utt_p, utt_pred_abs_p)
        per_generate = calc_error_rate(utt_p, utt_pred_generate_p)

        wer_gt_list.append(wer_gt)
        wer_abs_list.append(wer_abs)
        wer_generate_list.append(wer_generate)
        per_gt_list.append(per_gt)
        per_abs_list.append(per_abs)
        per_generate_list.append(per_generate)

        print(f'--- iter {i} ---')
        print(f'utt = {utt}')
        print(f'pesq_abs = {pesq_abs}')
        print(f'pesq_generate = {pesq_generate}')
        print(f'stoi_abs = {stoi_abs}')
        print(f'stoi_generate = {stoi_generate}')
        print(f'estoi_abs = {estoi_abs}')
        print(f'estoi_generate = {estoi_generate}')
        print(f'wer_gt = {wer_gt}')
        print(f'wer_abs = {wer_abs}')
        print(f'wer_generate = {wer_generate}')
        print(f'per_gt = {per_gt}')
        print(f'per_abs = {per_abs}')
        print(f'per_generate = {per_generate}')
        print('')

    pesq_abs = np.mean(pesq_abs_list)
    pesq_generate = np.mean(pesq_generate_list)
    stoi_abs = np.mean(stoi_abs_list)
    stoi_generate = np.mean(stoi_generate_list)
    estoi_abs = np.mean(estoi_abs_list)
    estoi_generate = np.mean(estoi_generate_list)
    wer_gt = np.mean(wer_gt_list)
    wer_abs = np.mean(wer_abs_list)
    wer_generate = np.mean(wer_generate_list)
    per_gt = np.mean(per_gt_list)
    per_abs = np.mean(per_abs_list)
    per_generate = np.mean(per_generate_list)

    file_name = save_path / f"{filename}.txt"
    with open(str(file_name), "a") as f:
        f.write("--- Objective Evaluation Metrics ---\n")
        f.write(f'speaker = {speaker}\n')
        f.write(f"pesq_abs = {pesq_abs:f}\n")
        f.write(f"pesq_generate = {pesq_generate:f}\n")
        f.write(f"stoi_abs = {stoi_abs:f}\n")
        f.write(f"stoi_generate = {stoi_generate:f}\n")
        f.write(f"estoi_abs = {estoi_abs:f}\n")
        f.write(f"estoi_generate = {estoi_generate:f}\n")
        f.write(f'wer_gt = {wer_gt * 100:f}%\n')
        f.write(f'wer_abs = {wer_abs * 100:f}%\n')
        f.write(f'wer_generate = {wer_generate * 100:f}%\n')
        f.write(f'per_gt = {per_gt * 100:f}%\n')
        f.write(f'per_abs = {per_abs * 100:f}%\n')
        f.write(f'per_generate = {per_generate * 100:f}%\n')
        f.write('\n')


def calc_accuracy_en(data_dir, save_path, cfg, filename):
    speaker = data_dir.stem
    wb_pesq_evaluator = PerceptualEvaluationSpeechQuality(cfg.model.sampling_rate, 'wb')
    stoi_evaluator = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, extended=False)
    estoi_evaluator = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, extended=True)
    speech_recognizer = whisper.load_model('large')
    utt_dir = Path(cfg.train.tcd_timit.text_dir).expanduser()

    pesq_abs_list = []
    pesq_generate_list = []
    stoi_abs_list = []
    stoi_generate_list = []
    estoi_abs_list = []
    estoi_generate_list = []
    wer_gt_list = []
    wer_abs_list = []
    wer_generate_list = []
    cer_gt_list = []
    cer_abs_list = []
    cer_generate_list = []

    gt_data_path_list = list(data_dir.glob('**/gt.wav'))
    for i, gt_data_path in enumerate(gt_data_path_list):
        abs_data_path = Path(str(gt_data_path).replace('gt', 'abs'))
        generate_data_path = Path(str(gt_data_path).replace('gt', 'generate'))
        wav_gt, _ = librosa.load(str(gt_data_path), sr=cfg.model.sampling_rate)
        wav_abs, _ = librosa.load(str(abs_data_path), sr=cfg.model.sampling_rate)
        wav_generate, _ = librosa.load(str(generate_data_path), sr=cfg.model.sampling_rate)
        min_sample = min(wav_gt.shape[0], wav_abs.shape[0], wav_generate.shape[0])
        wav_gt = wav_gt[:min_sample]
        wav_abs = wav_abs[:min_sample]
        wav_generate = wav_generate[:min_sample]
        wav_gt = torch.from_numpy(wav_gt)
        wav_abs = torch.from_numpy(wav_abs)
        wav_generate = torch.from_numpy(wav_generate)

        utt_path = utt_dir / speaker / 'straightcam' / f'{gt_data_path.parents[0].name}.txt'
        with open(str(utt_path), 'r') as f:
            utt = f.read().rstrip().replace(',', '').replace('.', '')

        pesq_abs = wb_pesq_evaluator(wav_abs, wav_gt)
        pesq_generate = wb_pesq_evaluator(wav_generate, wav_gt)
        stoi_abs = stoi_evaluator(wav_abs, wav_gt)
        stoi_generate = stoi_evaluator(wav_generate, wav_gt)
        estoi_abs = estoi_evaluator(wav_abs, wav_gt)
        estoi_generate = estoi_evaluator(wav_generate, wav_gt)
        pesq_abs_list.append(pesq_abs)
        pesq_generate_list.append(pesq_generate)
        stoi_abs_list.append(stoi_abs)
        stoi_generate_list.append(stoi_generate)
        estoi_abs_list.append(estoi_abs)
        estoi_generate_list.append(estoi_generate)

        utt_pred_gt = speech_recognizer.transcribe(str(gt_data_path), language='en')['text'].replace(',', '').replace('.', '')
        utt_pred_abs = speech_recognizer.transcribe(str(abs_data_path), language='en')['text'].replace(',', '').replace('.', '')
        utt_pred_generate = speech_recognizer.transcribe(str(generate_data_path), language='en')['text'].replace(',', '').replace('.', '')
        wer_gt = calc_error_rate(utt, utt_pred_gt)
        wer_abs = calc_error_rate(utt, utt_pred_abs)
        wer_generate = calc_error_rate(utt, utt_pred_generate)

        utt_c = [c for c in utt.replace(' ', '')]
        utt_pred_gt_c = [c for c in utt_pred_gt.replace(' ', '')]
        utt_pred_abs_c = [c for c in utt_pred_abs.replace(' ', '')]
        utt_pred_generate_c = [c for c in utt_pred_generate.replace(' ', '')]
        cer_gt = calc_error_rate(utt_c, utt_pred_gt_c)
        cer_abs = calc_error_rate(utt_c, utt_pred_abs_c)
        cer_generate = calc_error_rate(utt_c, utt_pred_generate_c)

        wer_gt_list.append(wer_gt)
        wer_abs_list.append(wer_abs)
        wer_generate_list.append(wer_generate)
        cer_gt_list.append(cer_gt)
        cer_abs_list.append(cer_abs)
        cer_generate_list.append(cer_generate)

        print(f'--- iter {i} ---')
        print(f'utt = {utt}')
        print(f'pesq_abs = {pesq_abs}')
        print(f'pesq_generate = {pesq_generate}')
        print(f'stoi_abs = {stoi_abs}')
        print(f'stoi_generate = {stoi_generate}')
        print(f'estoi_abs = {estoi_abs}')
        print(f'estoi_generate = {estoi_generate}')
        print(f'wer_gt = {wer_gt}')
        print(f'wer_abs = {wer_abs}')
        print(f'wer_generate = {wer_generate}')
        print(f'cer_gt = {cer_gt}')
        print(f'cer_abs = {cer_abs}')
        print(f'cer_generate = {cer_generate}')
        print('')
        
    pesq_abs = np.mean(pesq_abs_list)
    pesq_generate = np.mean(pesq_generate_list)
    stoi_abs = np.mean(stoi_abs_list)
    stoi_generate = np.mean(stoi_generate_list)
    estoi_abs = np.mean(estoi_abs_list)
    estoi_generate = np.mean(estoi_generate_list)
    wer_gt = np.mean(wer_gt_list)
    wer_abs = np.mean(wer_abs_list)
    wer_generate = np.mean(wer_generate_list)
    cer_gt = np.mean(cer_gt_list)
    cer_abs = np.mean(cer_abs_list)
    cer_generate = np.mean(cer_generate_list)

    file_name = save_path / f"{filename}.txt"
    with open(str(file_name), "a") as f:
        f.write("--- Objective Evaluation Metrics ---\n")
        f.write(f'speaker = {speaker}\n')
        f.write(f"pesq_abs = {pesq_abs:f}\n")
        f.write(f"pesq_generate = {pesq_generate:f}\n")
        f.write(f"stoi_abs = {stoi_abs:f}\n")
        f.write(f"stoi_generate = {stoi_generate:f}\n")
        f.write(f"estoi_abs = {estoi_abs:f}\n")
        f.write(f"estoi_generate = {estoi_generate:f}\n")
        f.write(f'wer_gt = {wer_gt * 100:f}%\n')
        f.write(f'wer_abs = {wer_abs * 100:f}%\n')
        f.write(f'wer_generate = {wer_generate * 100:f}%\n')
        f.write(f'cer_gt = {cer_gt * 100:f}%\n')
        f.write(f'cer_abs = {cer_abs * 100:f}%\n')
        f.write(f'cer_generate = {cer_generate * 100:f}%\n')
        f.write('\n')


def calc_accuracy(data_dir, save_path, cfg, filename, process_times=None):
    speaker = data_dir.stem
    wav2flac(data_dir)
    df = load_utt()

    wb_pesq = PerceptualEvaluationSpeechQuality(cfg.model.sampling_rate, 'wb')
    stoi = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, extended=False)
    estoi = ShortTimeObjectiveIntelligibility(cfg.model.sampling_rate, extended=True)
    r = sr.Recognizer()
    mecab = MeCab.Tagger('-Owakati')

    pesq_list = []
    stoi_list = []
    estoi_list = []
    duration = []
    rmse_power_list = []
    rmse_f0_list_librosa = []
    vuv_acc_list_librosa = []
    rmse_f0_list_world = []
    vuv_acc_list_world = []
    mcd_list = []
    wer_target_list = []
    wer_gen_list = []
    per_target_list = []
    per_gen_list = []
    iter_cnt = 0

    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                if abs_or_gen in Path(file).stem:
                    iter_cnt += 1
                    print(f"\niter_cnt : {iter_cnt}")
                    wav_gen, fs = torchaudio.load(os.path.join(curdir, file))
                    wav_in, fs = torchaudio.load(os.path.join(curdir, "input.wav"))
                    wav_gen = wav_gen.squeeze(0)
                    wav_in = wav_in.squeeze(0)

                    shorter_n_frame = int(min(wav_gen.shape[0], wav_in.shape[0]))
                    wav_gen = wav_gen[:shorter_n_frame]
                    wav_in = wav_in[:shorter_n_frame]
                    assert wav_gen.shape[0] == wav_in.shape[0]

                    # pesq, stoi, estoi
                    p = wb_pesq(wav_gen, wav_in)
                    s = stoi(wav_gen, wav_in)
                    es = estoi(wav_gen, wav_in)
                    pesq_list.append(p)
                    stoi_list.append(s)
                    estoi_list.append(es)
                    duration.append(shorter_n_frame / cfg.model.sampling_rate)
                    print(f"PESQ = {p}")
                    print(f"STOI = {s}")
                    print(f"ESTOI = {es}")

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

                    wav_gen = wav_gen.astype(np.float64)
                    wav_in = wav_in.astype(np.float64)

                    # rmse f0 & vuv accuracy by world
                    f0_gen, timeaxis_gen = pyworld.harvest(wav_gen, fs, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
                    ap_gen = pyworld.d4c(wav_gen, f0_gen, timeaxis_gen, fs, threshold=0.85)
                    vuv_flag_gen = (ap_gen[:, 0] < 0.5) * (f0_gen > 1.0)
                    vuv_gen = vuv_flag_gen.astype('int')
                    f0_in, timeaxis_in = pyworld.harvest(wav_in, fs, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
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

                    sp_gen = pyworld.cheaptrick(wav_gen, f0_gen, timeaxis_gen, fs)    
                    sp_in = pyworld.cheaptrick(wav_in, f0_in, timeaxis_in, fs)
                    alpha = pysptk.util.mcepalpha(fs)
                    mcep_gen = pysptk.mcep(sp_gen, order=cfg.model.mcep_order - 1, alpha=alpha, itype=4)    # cfg.model.mcep_order次元になる
                    mcep_in = pysptk.mcep(sp_in, order=cfg.model.mcep_order - 1, alpha=alpha, itype=4)
                    vuv_in = vuv_in[:, None]    # (T, 1)
                    vuv_in = np.repeat(vuv_in, mcep_gen.shape[1], axis=1)
                    mcep_gen = np.where(vuv_in == 1, mcep_gen, 0)   # 有声区間のみ
                    mcep_in = np.where(vuv_in == 1, mcep_in, 0)
                    mcd = melcd(mcep_gen, mcep_in)
                    mcd_list.append(mcd)
                    print(f"mcd = {mcd}")

                    # mfcc_gen = pyworld.code_spectral_envelope(sp_gen, fs, cfg.model.mcep_order)
                    # mfcc_in = pyworld.code_spectral_envelope(sp_in, fs, cfg.model.mcep_order)
                    # mcd = melcd(mfcc_gen, mfcc_in)
                    # print(f"mcd = {mcd}")

                    # mcd = melcd(mfcc_gen[:, :13], mfcc_in[:, :13])
                    # print(f"mcd = {mcd}")

                    # wer and per
                    for i in range(53):
                        utt_num = df[i][2]
                        if utt_num in Path(curdir).name:
                            utt = df[i][3]
                            utt = utt.replace("。", "").replace("、", "")

                    file_gen = Path(curdir, f"{abs_or_gen}.flac")
                    file_in = Path(curdir, "input.flac")

                    with sr.AudioFile(str(file_gen)) as source:
                        audio_gen = r.record(source)

                    with sr.AudioFile(str(file_in)) as source:
                        audio_in = r.record(source)

                    result_in = None
                    try:
                        result_in = r.recognize_google(audio_in, language="ja-JP")
                    except:
                        print("Recognizer can't understand what he or she is saying.")

                    result_gen = None
                    try:
                        result_gen = r.recognize_google(audio_gen, language="ja-JP")
                    except:
                        print("Recognizer can't understand what he or she is saying.")

                    if result_gen is not None and result_in is not None:
                        result_gen_w = mecab.parse(result_gen)
                        result_in_w = mecab.parse(result_in)
                        utt_w = mecab.parse(utt)
                        error_in = wer(utt_w, result_in_w)
                        error_gen = wer(utt_w, result_gen_w)
                        print(f"target : {utt_w}")
                        print(f"gen : {result_gen_w}")
                        print(f"ref : {result_in_w}")
                        print(f"wer_ref = {error_in:f}, wer_gen = {error_gen:f}")
                        wer_target_list.append(error_in)
                        wer_gen_list.append(error_gen)

                        result_gen_p = pyopenjtalk.g2p(result_gen)
                        result_in_p = pyopenjtalk.g2p(result_in)
                        utt_p = pyopenjtalk.g2p(utt)
                        error_in = wer(utt_p, result_in_p)
                        error_gen = wer(utt_p, result_gen_p)
                        print(f"target : {utt_p}")
                        print(f"gen : {result_gen_p}")
                        print(f"ref : {result_in_p}")
                        print(f"per_ref = {error_in:f}, per_gen = {error_gen:f}")
                        per_target_list.append(error_in)
                        per_gen_list.append(error_gen)
                    else:
                        result_in_w = mecab.parse(result_in)
                        utt_w = mecab.parse(utt)
                        error_in = wer(utt_w, result_in_w)
                        wer_target_list.append(error_in)
                        wer_gen_list.append(1)

                        result_in_p = pyopenjtalk.g2p(result_in)
                        utt_p = pyopenjtalk.g2p(utt)
                        error_in = wer(utt_p, result_in_p)
                        per_target_list.append(error_in)
                        per_gen_list.append(1)

        if cfg.test.debug:
            if iter_cnt > 2:
                break

    if debug == False:
        pesq = sum(pesq_list) / len(pesq_list)
        stoi = sum(stoi_list) / len(stoi_list)
        estoi = sum(estoi_list) / len(estoi_list)
        rmse_power = sum(rmse_power_list) / len(rmse_power_list)
        rmse_f0_librosa = sum(rmse_f0_list_librosa) / len(rmse_f0_list_librosa)
        vuv_acc_librosa = sum(vuv_acc_list_librosa) / len(vuv_acc_list_librosa)
        rmse_f0_world = sum(rmse_f0_list_world) / len(rmse_f0_list_world)
        vuv_acc_world = sum(vuv_acc_list_world) / len(vuv_acc_list_world)
        mcd = sum(mcd_list) / len(mcd_list)
        wer_target = sum(wer_target_list) / len(wer_target_list)
        wer_gen = sum(wer_gen_list) / len(wer_gen_list)
        per_target = sum(per_target_list) / len(per_target_list)
        per_gen = sum(per_gen_list) / len(per_gen_list)

        file_name = save_path / f"{filename}.txt"
        with open(str(file_name), "a") as f:
            f.write("--- Objective Evaluation Metrics ---\n")
            f.write(f'speaker = {speaker}\n')
            f.write(f"PESQ = {pesq:f}\n")
            f.write(f"STOI = {stoi:f}\n")
            f.write(f"ESTOI = {estoi:f}\n")
            f.write(f"rmse power = {rmse_power:f}dB\n")
            f.write(f"rmsef0 librosa = {rmse_f0_librosa:f}\n")
            f.write(f"vuv accuracy librosa = {vuv_acc_librosa:f}%\n")
            f.write(f"rmse f0 world = {rmse_f0_world:f}\n")
            f.write(f"vuv accuracy world = {vuv_acc_world:f}%\n")
            f.write(f"mel cepstral distortion = {mcd:f}dB\n")
            f.write(f"word error rate target = {wer_target * 100:f}%\n")
            f.write(f"word error rate gen = {wer_gen * 100:f}%\n")
            f.write(f"phoneme error rate target = {per_target * 100:f}%\n")
            f.write(f"phoneme error rate gen = {per_gen * 100:f}%\n")

            if process_times is not None:
                f.write("\n--- Duration and Process Time ---\n")
                f.write(f"duration_mean = {sum(duration) / len(duration):f}, process_time_mean = {sum(process_times) / len(process_times):f}\n")
                for dur, time in zip(duration, process_times):
                    f.write(f"duration = {dur:f}, process_time = {time:f}\n")

            f.write('\n')
                    

def calc_mean(result_file_path):
    with open(str(result_file_path), 'r') as f:
        content = f.readlines()

    result_dict = defaultdict(float)
    cnt = 0
    for line in content:
        key = line.strip().split('=')[0][:-1]
        if key == 'speaker':
            cnt += 1
        value = re.findall(r'\d+\.\d+', line)
        if value:
            value = [float(v) for v in value][0]
            result_dict[key] += value

    result_dict = {key: value / cnt for key, value in result_dict.items()}
    with open(str(result_file_path), 'a') as f:
        f.write('--- mean ---\n')
        for key, value in result_dict.items():
            f.write(f'{key} = {value}\n')
        f.write('\n')