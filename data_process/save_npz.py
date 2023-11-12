from pathlib import Path
import sys
import hydra
import librosa
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))
from data_process.feature import wav2mel


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    data_dir = Path('/home/minami/dataset/lip/wav/F01_kablab')
    data_path_list = data_dir.glob('*.wav')
    save_dir = Path('/home/minami/dataset/lip/mel_npz/F01_kablab')
    for data_path in tqdm(data_path_list):
        wav, _ = librosa.load(str(data_path), sr=cfg.model.sampling_rate)
        mel = wav2mel(wav, cfg, ref_max=False)
        save_path = save_dir / f'{data_path.stem}'
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        np.savez(
            file=str(save_path),
            mel=mel,
        )
        


if __name__ == '__main__':
    main()