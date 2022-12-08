"""
口唇部分の切り抜きをおこなった後,croppedからwavだけコピーするために使用する
F01_kablabについて,norm.wavがよくわからないので,そこは無視して普通のwavだけをコピーしてます
使用時はspeakerを変更してください
"""

import shutil
import os
from pathlib import Path
from tqdm import tqdm


speaker = "M04_kablab"
margin = 0
fps = 50
audio_path = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
save_path = Path(f"~/dataset/lip/face_cropped_nn_{margin}_{fps}/{speaker}").expanduser()


def main():
    os.makedirs(save_path, exist_ok=True)
    print(f"speaker = {speaker}")
    wavs = []
    for curdir, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.wav'):
                if 'norm' in Path(file).stem:
                    # normを回避
                    continue
                else:
                    wavs.append(os.path.join(curdir, file))
    
    for i in tqdm(range(len(wavs))):
        shutil.copy(wavs[i], save_path)


if __name__ == "__main__":
    main()

