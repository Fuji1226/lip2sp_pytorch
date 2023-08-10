from pathlib import Path
import os
import numpy as np
import re
import av


DEBUG = False
SR = 16000
NUM_FRAME_LIMIT_MIN = 50

LANDMARK_SAVE_DIR_NAME = "landmark_split_wide"
BBOX_SAVE_DIR_NAME = "bbox_split_wide"
VIDEO_SAVE_DIR_NAME = "Dataset_split_wide"

if DEBUG:
    LANDMARK_SAVE_DIR_NAME = LANDMARK_SAVE_DIR_NAME + "_debug"
    BBOX_SAVE_DIR_NAME = BBOX_SAVE_DIR_NAME + "_debug"
    VIDEO_SAVE_DIR_NAME = VIDEO_SAVE_DIR_NAME + "_debug"


def main():
    data_dir = Path("~/Lip2Wav/Dataset_fps25").expanduser()
    splitted_data_dir = Path(str(data_dir).replace("Dataset", VIDEO_SAVE_DIR_NAME))
    speaker_list = ["chem", "chess", "dl", "eh", "hs"]
    for speaker in speaker_list:
        spk_dir = data_dir / speaker / "intervals"
        data_path_list = list(spk_dir.glob("*/*.mp4"))
        
        splitted_data_dir_spk = splitted_data_dir / speaker
        splitted_data_path_list = list(splitted_data_dir_spk.glob("*/*/*.mp4"))
        splitted_data_path_list = [f"{data_path.parents[0].name}/{str(data_path.stem).split('_')[0]}" for data_path in splitted_data_path_list]
        
        for data_path in data_path_list:
            breakpoint()
    
    
if __name__ == "__main__":
    main()