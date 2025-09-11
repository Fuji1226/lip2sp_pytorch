from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os

def make_spk_emb_jvs(spk_dir, out_dir):
    encoder = VoiceEncoder()
    spk_dir = Path(spk_dir)
    wav_dir = Path(spk_dir / "parallel100/wav24kHz16bit")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_dir.glob("*.wav"):
        try:
            wav = preprocess_wav(wav_path)
            emb = encoder.embed_utterance(wav)  # (256,)
            spk_id = spk_dir.name  # e.g. jvs019
            np.save(out_dir / f"{spk_id}.npy", emb)
            print(f"Extracted: {spk_id}")
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")



def make_spk_emb_katsurada(spk_dir, out_dir):
    encoder = VoiceEncoder()
    spk_dir = Path(spk_dir)
    wav_dir = Path(spk_dir / "audio/train")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_dir.glob("*.wav"):
        try:
            wav = preprocess_wav(wav_path)
            emb = encoder.embed_utterance(wav)  # (256,)
            spk_id = spk_dir.name  # e.g. jvs019
            np.save(out_dir / f"{spk_id}.npy", emb)
            print(f"Extracted: {spk_id}")
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")


def make_spk_emb_kab2022(spk_dir, out_dir):
    encoder = VoiceEncoder()
    spk_dir = Path(spk_dir)
    wav_dir = Path(spk_dir / "wav")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_dir.glob("*.wav"):
        try:
            wav = preprocess_wav(wav_path)
            emb = encoder.embed_utterance(wav)  # (256,)
            spk_id = spk_dir.name  # e.g. jvs019
            np.save(out_dir / f"{spk_id}.npy", emb)
            print(f"Extracted: {spk_id}")
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")



"""
#?jvsのやつ実行する↓
for i in range(1,101) : #話者ごとにやる

    spk = f"jvs{i:03d}"

    spk_dir = Path('/home/user/dataset/jvs_ver1/')/str(spk)
    make_spk_emb_jvs(spk_dir, '/home/user/dataset/jvs_ver1/emb')
"""
"""
#?katsuradaのやつ実行する↓
spk_dir = '/home/user/2HEAVD/ITA_text/F1'
out_dir = '/home/user/2HEAVD/ITA_text/F1/audio/train_emb'
make_spk_emb_katsurada(spk_dir, out_dir)
"""

#?kab2022のやつ実行する↓
spk_dir = '/home/user/dataset/M02_kablab'
out_dir = '/home/user/dataset/lip/emb/M02_kablab'
make_spk_emb_kab2022(spk_dir, out_dir)