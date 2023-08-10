import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/data_process").expanduser()))

import pandas as pd
import numpy as np
import hydra
from tqdm import tqdm

from transform import load_data_lrs2
from face_crop_align import FaceAligner


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg):
    data_dir = Path("~/Lip2Wav").expanduser()
    save_dir = Path("~/dataset/lip/np_files/lip2wav/train").expanduser()
    bbox_dir = data_dir / "bbox_split_wide_fps25"
    speaker_list = ['hs', 'eh']
    
    desired_left_eye = (cfg.model.align_desired_left_eye, cfg.model.align_desired_left_eye)
    desired_face_size = cfg.model.align_desired_face_size
    aligner = FaceAligner(desired_left_eye, desired_face_size, desired_face_size)

    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        bbox_dir_spk = bbox_dir / speaker / "intervals"
        bbox_path_list = list(bbox_dir_spk.glob("*/*.csv"))
        
        for bbox_path in tqdm(bbox_path_list):
            video_path = Path(str(bbox_path).replace("bbox", "Dataset").replace(".csv", "_merged.mp4"))
            landmark_path = Path(str(bbox_path).replace("bbox", "landmark"))

            save_path = save_dir / speaker / "mspec80"
            filename = "_".join([video_path.parents[0].name, bbox_path.stem])

            if (save_path / f'{filename}.npz').exists():
                continue
            
            # wav, lip, feature, data_len = load_data_lrs2(video_path, bbox_path, landmark_path, cfg, aligner)
            # save_path.mkdir(parents=True, exist_ok=True)
            # np.savez(
            #     str(save_path / filename),
            #     wav=wav,
            #     lip=lip,
            #     feature=feature,
            # )
            
            try:
                wav, lip, feature, data_len = load_data_lrs2(video_path, bbox_path, landmark_path, cfg, aligner)
                save_path = save_dir / speaker / "mspec80"
                save_path.mkdir(parents=True, exist_ok=True)
                filename = "_".join([video_path.parents[0].name, bbox_path.stem])
                np.savez(
                    str(save_path / filename),
                    wav=wav,
                    lip=lip,
                    feature=feature,
                )
            except:
                print(video_path)
                continue
                
                
if __name__ == "__main__":
    main()