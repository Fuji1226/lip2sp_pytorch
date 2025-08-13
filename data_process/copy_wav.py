import shutil
import os
from pathlib import Path
from tqdm import tqdm





def pre_script():
    speaker = "M04_kablab"
    audio_path = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
    save_path = Path(f"~/dataset/lip/cropped_max_size_fps25/{speaker}").expanduser()
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

# 元ディレクトリ(/media/user/TOSHIBA_EXT/20220930/{numbers}/mov/label/splitted内のmp4ファイルを、~/dataset/kab2022/movにコピーする
def copy_mov_to_pc():
    src_dir = Path("/media/user/TOSHIBA_EXT/20220930")
    dst_dir = Path("~/dataset/kab2022/mov").expanduser()
    os.makedirs(dst_dir, exist_ok=True)

    for number in range(11, 13):  # 11から12までの数字を想定
        number = "930_" + str(number)  # 930_11, 930_12の形式に変換
        mov_dir = src_dir / str(number) / "mov" / "label" / "splitted"
        if mov_dir.exists():
            for file in mov_dir.glob("*.mp4"):
                shutil.copy(file, dst_dir)
                print(f"Copied {file} to {dst_dir}")

# copy_wav_to_pc関数を、copy_mov_to_pcと同様に作成
def copy_wav_to_pc():
    src_dir = Path("/media/user/TOSHIBA_EXT/20220930")
    dst_dir = Path("~/dataset/kab2022/wav").expanduser()
    os.makedirs(dst_dir, exist_ok=True)

    for number in range(11, 13):  # 11から12までの数字を想定
        number = "930_" + str(number)  # 930_11, 930_12の形式に変換
        wav_dir = src_dir / str(number) / "wav" / "wav_pad"
        if wav_dir.exists():
            for file in wav_dir.glob("*.wav"):
                shutil.copy(file, dst_dir)
                print(f"Copied {file} to {dst_dir}")

#コメントアウトで実行する処理の管理
if __name__ == "__main__":
    #pre_script()
    copy_mov_to_pc()
    copy_wav_to_pc()
