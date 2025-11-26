
import os
from pathlib import Path
import numpy as np
import torch


def select_checkpoint(path, key):
    '''
    checkpointの中から最も検証データに対しての損失が小さいものを選ぶ
    '''
    checkpoint_path_last = Path(path).expanduser()
    checkpoint_dict_last = torch.load(str(checkpoint_path_last))
    #print(f"使えるキー: {list(checkpoint_dict_last.keys())}")
    best_checkpoint = np.argmin(checkpoint_dict_last[key]) + 1
    filename_prev = checkpoint_path_last.stem + checkpoint_path_last.suffix
    filename_new = str(best_checkpoint) + checkpoint_path_last.suffix
    checkpoint_path = Path(str(checkpoint_path_last).replace(filename_prev, filename_new))
    return checkpoint_path


path = '/home/user/lip2sp_pytorch/check_point/nar/large/avhubert_preprocess_fps25_gray/master/Large_FT_pre/30.ckpt'
key = "val_mae_loss_list"

use_cp_path = select_checkpoint(path,key)
print(f"use checkpoint path: {use_cp_path}")
#breakpoint()
    #use_cp_path と同じディレクトリにある他のcheckpointファイルを削除する
cp_dir = use_cp_path.parents[0]
cp_list = list(cp_dir.glob('*'))
for cp in cp_list:
    if cp == use_cp_path:
        continue
    os.remove(str(cp))
