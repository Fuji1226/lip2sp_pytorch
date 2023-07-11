import pandas as pd
from pathlib import Path

n_data_use = 200
train_df_path = Path("~/lrs2/train.txt")
train_data_df = pd.read_csv(str(train_df_path), header=None)
train_data_df = train_data_df.rename(columns={0: "filename_all"})
train_data_df["id"] = train_data_df["filename_all"].apply(lambda x: str(x.split("/")[0]))
train_data_id_used = train_data_df["id"].value_counts().nlargest(n_data_use).index.values
train_data_df = train_data_df.loc[train_data_df["id"].isin(train_data_id_used)]

data_dir = Path("~/lrs2/mvlrs_v1/main").expanduser()
data_path_list = train_data_df["filename_all"].values
for data_path in data_path_list:
    data_path = data_dir / f"{data_path}.mp4"
    

breakpoint()