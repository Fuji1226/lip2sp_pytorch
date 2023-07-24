from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd

data_dir = Path("~/lrs2/mvlrs_v1").expanduser()
dirname_list = ["main", "pretrain"]
save_dir = Path("~/lrs2").expanduser()

for dirname in dirname_list:
    data_dir_ = data_dir / dirname
    spk_list = list(data_dir_.glob("*"))
    info_list = []
    
    for spk in tqdm(spk_list):
        data_path_list = list(spk.glob("*.mp4"))
        info_list.append([str(spk.stem), len(data_path_list)])
        
    info_df = pd.DataFrame(info_list, columns=["id", "n_data"])
    info_df.to_csv(str(save_dir / f"data_info_{dirname}.csv"), index=False)