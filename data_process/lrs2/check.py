import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2


def check_duration(file_list, data_dir, data_df):
    for filename in tqdm(file_list):
        data_path = data_dir / f"{filename}.mp4"
        cap = cv2.VideoCapture(str(data_path))
        video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_len_sec = video_frame_count / video_fps
        data_df.loc[data_df["filename_all"] == filename, "duration"] = video_len_sec
    return data_df
        

def main():
    n_largest_sampling = 200
    
    train_df_path = Path("~/lrs2/train.txt")
    train_data_df = pd.read_csv(str(train_df_path), header=None)
    train_data_df = train_data_df.rename(columns={0: "filename_all"})
    train_data_df["id"] = train_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
    train_id_list = train_data_df["id"].value_counts().sort_values().tail(n_largest_sampling).index.unique().to_list()
    train_data_df = train_data_df.loc[train_data_df["id"].isin(train_id_list)]
    train_file_list = train_data_df["filename_all"].to_list()
    
    pretrain_df_path = Path("~/lrs2/pretrain.txt")
    pretrain_data_df = pd.read_csv(str(pretrain_df_path), header=None)
    pretrain_data_df = pretrain_data_df.rename(columns={0: "filename_all"})
    pretrain_data_df["id"] = pretrain_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
    pretrain_id_list = pretrain_data_df["id"].value_counts().sort_values().tail(n_largest_sampling).index.unique().to_list()
    pretrain_data_df = pretrain_data_df.loc[pretrain_data_df["id"].isin(pretrain_id_list)]
    pretrain_file_list = pretrain_data_df["filename_all"].to_list()
    
    train_dir = Path("~/lrs2/mvlrs_v1/main").expanduser()
    pretrain_dir = Path("~/lrs2/mvlrs_v1/pretrain").expanduser()
    
    print("start main")
    train_data_df = check_duration(train_file_list, train_dir, train_data_df)
    
    print("start pretrain")
    pretrain_data_df = check_duration(pretrain_file_list, pretrain_dir, pretrain_data_df)
    
    save_dir = Path("~/lip2sp_pytorch/data_process/lrs2/duration_check").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    train_data_df.to_csv(str(save_dir / "main.csv"), index=False)
    pretrain_data_df.to_csv(str(save_dir / "pretrain.csv"), index=False)
    
    breakpoint()


if __name__ == "__main__":
    main()