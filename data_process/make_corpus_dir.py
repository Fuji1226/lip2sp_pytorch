"""
lip_croppedを作った後に,それらをコーパスごとのディレクトリに分離する
"""
from pathlib import Path
import os
import shutil
from tqdm import tqdm

debug = False

speaker = "M04_kablab"
corpus = ["ATR", "balanced", "BASIC5000"]
data_dir = Path(f"~/dataset/lip/lip_cropped/{speaker}").expanduser()

def main():
    print(f"speaker = {speaker}")
    for co in corpus:
        print(f"processing {co}")
        corpus_dir = data_dir / co
        os.makedirs(str(corpus_dir), exist_ok=True)
        data_path = sorted(list(data_dir.glob(f"{co}*.*")))

        for path in tqdm(data_path):
            shutil.move(str(path), str(corpus_dir))

            if debug:
                break
        

if __name__ == "__main__":
    main()
