from pathlib import Path
from tqdm import tqdm
import numpy as np
import hydra
from feature import wav2mel, wav2mel_avhubert


def pad(feature, data_len):
    feature_padded = np.zeros((feature.shape[0], data_len), dtype=feature.dtype)
    feature_padded[:, :feature.shape[-1]] = feature
    return feature_padded


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    data_dir = Path('/home/minami/dataset/lip/np_files/jsut')
    data_path_list = list(data_dir.glob('**/*.npz'))
    for data_path in tqdm(data_path_list):
        npz_key = np.load(str(data_path))
        lip = npz_key['lip']
        wav = npz_key['wav']
        feature = wav2mel(wav, cfg, ref_max=False)
        feature_avhubert = wav2mel_avhubert(wav, cfg)
        mul_factor = (cfg.model.sampling_rate // cfg.model.hop_length) // cfg.model.fps
        data_len = int(lip.shape[-1] * mul_factor)
        feature = feature[:, :data_len]
        feature_avhubert = feature_avhubert[:, :data_len]
        if feature.shape[-1] != data_len:
            feature = pad(feature, data_len)
        if feature_avhubert.shape[-1] != data_len:
            feature_avhubert = pad(feature_avhubert, data_len)
        assert feature.shape[-1] == data_len
        assert feature_avhubert.shape[-1] == data_len
        
        save_path = Path(str(data_path).replace('mspec80', 'mspec_avhubert'))
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        np.savez(
            str(save_path.parents[0] / save_path.stem),
            wav=wav,
            lip=lip,
            feature=feature.T,
            feature_avhubert=feature_avhubert.T,
        )


if __name__ == '__main__':
    main()