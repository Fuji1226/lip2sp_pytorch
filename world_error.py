#音声を読み込んで、world特徴量で比べて分析する
import os
import numpy as np
import librosa
import pyworld as pw
import soundfile as sf

from data_process.feature import wav2world

def world_output(wav_input,wav_AbS,wav_gen,cfg):
        # 音声波形からWORLD特徴量を計算
    spec_input, f0_input, vuv_input, ap_input = wav2world(
        wave=wav_input,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        cfg=cfg,
        sp_type="spec",
        plot=True,
    )
    spec_AbS, f0_AbS, vuv_AbS, ap_AbS = wav2world(
        wave=wav_AbS,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        cfg=cfg,
        sp_type="spec",
        plot=True,
    )
    spec_gen, f0_gen, vuv_gen, ap_gen = wav2world(
        wave=wav_gen,
        fs=cfg.model.sampling_rate,
        frame_period=cfg.model.frame_period,
        cfg=cfg,
        sp_type="spec",
        plot=True,
    )

    # safe MAE: 0 または NaN の箇所はスキップする
    # 長さが異なる可能性があれば短い方に合わせる
    min_len = min(f0_input.shape[0], f0_AbS.shape[0], f0_gen.shape[0])
    f0_in = f0_input[:min_len]
    f0_ab = f0_AbS[:min_len]
    f0_ge = f0_gen[:min_len]

    #AbS-Ref f0mae
    valid_mask = (~np.isclose(f0_ab, 0.0)) & (~np.isclose(f0_in, 0.0)) & (~np.isnan(f0_ab)) & (~np.isnan(f0_in))
    if np.any(valid_mask):
        f0_mae_AbS = float(np.mean(np.abs(f0_in[valid_mask] - f0_ab[valid_mask])))
    else:
        f0_mae_AbS = float('nan')  # 有効値が無ければ NaN を返す（必要なら 0.0 に変更）

    #Gen-Ref f0mae
    valid_mask = (~np.isclose(f0_in, 0.0)) & (~np.isclose(f0_ge, 0.0)) & (~np.isnan(f0_in)) &  (~np.isnan(f0_ge))
    if np.any(valid_mask):
        f0_mae_gen = float(np.mean(np.abs(f0_in[valid_mask] - f0_ge[valid_mask])))
    else:
        f0_mae_gen = float('nan')  # 有効値が無ければ NaN を返す（必要なら 0.0 に変更）

    return f0_mae_AbS, f0_mae_gen

    #TODO:f0_shift_mae(サンプル間のピッチシフトの差のmae)でやれたら嬉しい、Nan区間をどうする
    #TODO:vuvについても何かしらしたいかも