import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2


def main():
    N_DATA_USE = 200
    train_df_path = Path("~/lrs2/pretrain.txt")
    train_data_df = pd.read_csv(str(train_df_path), header=None)
    train_data_df = train_data_df.rename(columns={0: "filename_all"})
    train_data_df["id"] = train_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
    train_data_id_used = train_data_df["id"].value_counts().nlargest(N_DATA_USE).index.values
    train_data_df = train_data_df.loc[train_data_df["id"].isin(train_data_id_used)]


if __name__ == "__main__":
    main()