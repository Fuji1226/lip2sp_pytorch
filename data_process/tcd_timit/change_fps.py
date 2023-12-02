import cv2
from pathlib import Path
from tqdm import tqdm
import subprocess


def main():
    data_dir = Path('~/tcd_timit').expanduser()
    data_path_list = list(data_dir.glob('**/*.mp4'))
    for data_path in tqdm(data_path_list):
        save_path = Path(str(data_path).replace('tcd_timit', 'tcd_timit_fps25'))
        if save_path.exists():
            continue
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                'ffmpeg',
                '-i',
                str(data_path),
                '-r',
                '25',
                str(save_path)
            ]
        )


if __name__ == '__main__':
    main()