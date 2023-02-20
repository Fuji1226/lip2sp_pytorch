from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import csv
import shutil

def main():
    dirname_list = [
        "cropped_fps25", "cropped", "bbox", "bbox_aligned", "bbox_fps25", 
        "face_aligned", "landmark", "landmark_aligned", "landmark_fps25",
    ]
    speaker_list = ["F01_kablab_20220930", "F01_kablab_all"]
    for dirname in dirname_list:
        for speaker in speaker_list:
            data_dir = Path("~/dataset/lip").expanduser()
            data_dir = data_dir / dirname / speaker
            data_path = list(data_dir.glob("*"))
            for path in tqdm(data_path):
                if "BASIC5000_BASIC5000_" in path.stem:
                    prename = path.stem
                    postname = str(path.stem).split("_")[1:]
                    postname = "_".join(postname)
                    source_path = path.parents[0] / f"{prename}{path.suffix}"
                    dist_path = path.parents[0] / f"{postname}{path.suffix}"
                    os.rename(source_path, dist_path)


if __name__ == "__main__":
    main()