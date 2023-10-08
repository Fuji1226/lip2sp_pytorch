import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import hydra
import librosa
sys.path.append(str(Path('~/lip2sp_pytorch/data_process').expanduser()))

from feature import wav2mel, wav2mel_avhubert
from transform import get_upsample


def pad(x, data_len):
    x_padded = np.zeros((x.shape[0], data_len), dtype=x.dtype)
    x_padded[:, :x.shape[-1]] = x
    return x_padded


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg):
    speaker = 'male'
    data_list = ['train_parallel', 'train_non_parallel', 'dev', 'eval']
    save_dir = Path('~/dataset/lip/np_files/hifi-captain').expanduser()

    for data in data_list:
        data_dir = Path(f'~/dataset/hi-fi-captain/ja-JP').expanduser() / speaker / 'wav' / data
        data_path_list = list(data_dir.glob('*.wav'))

        for data_path in tqdm(data_path_list):
            wav, fs = librosa.load(str(data_path), sr=cfg.model.sampling_rate, mono=None)
            wav = wav / np.max(np.abs(wav), axis=0)
            upsample = get_upsample(cfg)
            feature = wav2mel(wav, cfg, ref_max=False)
            feature_avhubert = wav2mel_avhubert(wav, cfg)

            data_len = min(feature.shape[-1] // upsample * upsample, feature_avhubert.shape[-1] // upsample * upsample)
            feature = feature[:, :data_len]
            feature_avhubert = feature_avhubert[:, :data_len]
            if feature.shape[-1] != data_len:
                feature = pad(feature, data_len)
            if feature_avhubert.shape[-1] != data_len:
                feature_avhubert = pad(feature_avhubert, data_len)            
            assert feature.shape[-1] == data_len
            assert feature_avhubert.shape[-1] == data_len

            wav = wav[:data_len * cfg.model.hop_length]
            wav_padded = np.zeros(int(data_len * cfg.model.hop_length))
            wav_padded[:wav.shape[0]] = wav
            wav = wav_padded
            lip = np.random.rand(1, 96, 96, int(data_len * upsample))
            
            if data == 'train_parallel' or data == 'train_non_parallel':
                save_path = save_dir / 'train'
            elif data == 'dev':
                save_path = save_dir / 'val'
            elif data =='eval':
                save_path = save_dir / 'test'
            save_path = save_path / speaker / cfg.model.name / data_path.stem
            save_path.parents[0].mkdir(parents=True, exist_ok=True)
            np.savez(
                str(save_path),
                wav=wav,
                feature=feature,
                feature_avhubert=feature_avhubert,
            )


if __name__ == '__main__':
    main()