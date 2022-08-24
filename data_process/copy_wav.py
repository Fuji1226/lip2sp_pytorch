"""
口唇部分の切り抜きをおこなった後,croppedからwavだけコピーするために使用する
norm.wavがよくわからないので,そこは無視して普通のwavだけをコピーしてます

使用するときはaudio_path(wavファイルがあるディレクトリ)とsave_pathを変更してください
"""

import shutil
import os
from pathlib import Path

def main():
    audio_path = "/home/usr4/r70264c/dataset/lip/cropped/F01_kablab"
    save_path = "/home/usr4/r70264c/dataset/lip/lip_cropped/F01_kablab"
    wavs = []
    for curdir, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.wav'):
                if 'norm' in Path(file).stem:
                    # normを回避
                    continue
                else:
                    wavs.append(os.path.join(curdir, file))

    for i in range(len(wavs)):
        shutil.copy(wavs[i], save_path)


if __name__ == "__main__":
    main()

