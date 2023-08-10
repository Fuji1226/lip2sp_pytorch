from pathlib import Path
import os
import numpy as np
import re
import av
import cv2
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from librosa.display import specshow


def main():
    speaker = 'hs'

    save_dir = Path(f'~/lip2sp_pytorch/data_process/lip2wav/npz_check/{speaker}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(f'~/dataset/lip/np_files/lip2wav/train/{speaker}/mspec80').expanduser()
    data_path_list = list(data_dir.glob('*.npz'))

    data_path_list = data_path_list[:5]

    for data_path in data_path_list:
        npz_key = np.load(str(data_path))
        lip = npz_key['lip']
        wav = npz_key['wav']
        feature = npz_key['feature'].T

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            filename=str(save_dir / f'{data_path.stem}.mp4'), 
            fourcc=fourcc, 
            fps=25.0, 
            frameSize=(lip.shape[1], lip.shape[2]), 
            isColor=False
        )
        lip = np.transpose(lip, (1, 2, 0, 3))   # (H, W, 3, T)
        for i in range(lip.shape[-1]):
            frame = lip[..., i]
            out.write(frame)
        out.release()

        write(str(save_dir / f'{data_path.stem}.wav'), rate=16000, data=wav)

        plt.figure(figsize=(6, 4))
        specshow(
            data=feature, 
            x_axis="time", 
            y_axis="mel", 
            sr=16000, 
            hop_length=160,
            fmin=0,
            fmax=8000,
            cmap="viridis",
        )
        plt.colorbar(format="%+2.f dB")
        plt.xlabel("Time[s]")
        plt.ylabel("Frequency[Hz]")
        plt.savefig(str(save_dir / f'{data_path.stem}_mel.png'))
        plt.close()

        plt.figure()
        time = np.arange(wav.shape[0]) / 16000
        plt.plot(time, wav)
        plt.grid()
        plt.savefig(str(save_dir / f'{data_path.stem}_waveform.png'))
        plt.close()

    
    
if __name__ == "__main__":
    main()