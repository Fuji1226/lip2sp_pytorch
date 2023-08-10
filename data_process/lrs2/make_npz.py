import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/data_process").expanduser()))

import pandas as pd
import numpy as np
import hydra
from tqdm import tqdm
import argparse

from transform import load_data_lrs2
from face_crop_align import FaceAligner


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--data")
    argParser.add_argument("--num")
    args = argParser.parse_args()
    
    if args.data == "main":
        train_df_path = Path("~/lrs2/train.txt")
        train_data_df = pd.read_csv(str(train_df_path), header=None)
        train_data_df = train_data_df.rename(columns={0: "filename_all"})
        train_data_df["id"] = train_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
        train_data_id_used = train_data_df["id"].value_counts().nlargest(args.num).index.values
        train_data_df = train_data_df.loc[train_data_df["id"].isin(train_data_id_used)]
        data_dir = Path("~/lrs2/mvlrs_v1/main").expanduser()
        landmark_dir = Path("~/lrs2/landmark/main").expanduser()
        bbox_dir = Path("~/lrs2/bbox/main").expanduser()
        save_dir = Path("~/dataset/lip/np_files/lrs2/train").expanduser()
        
    elif args.data == "pretrain":
        train_df_path = Path("~/lrs2/pretrain.txt")
        train_data_df = pd.read_csv(str(train_df_path), header=None)
        train_data_df = train_data_df.rename(columns={0: "filename_all"})
        train_data_df["id"] = train_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
        train_data_id_used = train_data_df["id"].value_counts().nlargest(args.num).index.values
        train_data_df = train_data_df.loc[train_data_df["id"].isin(train_data_id_used)]
        data_dir = Path("~/lrs2/mvlrs_v1/pretrain").expanduser()
        landmark_dir = Path("~/lrs2/landmark/pretrain").expanduser()
        bbox_dir = Path("~/lrs2/bbox/pretrain").expanduser()
        save_dir = Path("~/dataset/lip/np_files/lrs2_pretrain/train").expanduser()
    
    desired_left_eye = (cfg.model.align_desired_left_eye, cfg.model.align_desired_left_eye)
    desired_face_size = cfg.model.align_desired_face_size
    aligner = FaceAligner(desired_left_eye, desired_face_size, desired_face_size)
    data_path_list = train_data_df["filename_all"].values
    
    for data_path in tqdm(data_path_list):
        video_path = data_dir / f"{data_path}.mp4"
        landmark_path = landmark_dir / f"{data_path}.csv"
        bbox_path = bbox_dir / f"{data_path}.csv"
        speaker = video_path.parents[0].name
        filename = video_path.stem
        
        try:
            wav, lip, feature, data_len = load_data_lrs2(video_path, bbox_path, landmark_path, cfg, aligner)
            save_path = save_dir / speaker / "mspec80"
            save_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(save_path / filename),
                wav=wav,
                lip=lip,
                feature=feature,
            )
        except:
            print(data_path)
            continue
        
        
        
if __name__ == "__main__":
    main()