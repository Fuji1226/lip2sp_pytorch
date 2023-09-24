import pandas as pd
import os
import shutil
from pathlib import Path

# save_csv.pyで決めた分け方でファイル分割
def main():
    df = pd.read_csv('/home/minami/dataset/jsut_ver1.1/filename.csv')
    npz_dir_train = Path('/home/minami/dataset/lip/np_files/jsut/train/female/mspec80')
    npz_dir_val = Path('/home/minami/dataset/lip/np_files/jsut/val/female/mspec80')
    npz_dir_test = Path('/home/minami/dataset/lip/np_files/jsut/test/female/mspec80')
    npz_dir_train.mkdir(parents=True, exist_ok=True)
    npz_dir_val.mkdir(parents=True, exist_ok=True)
    npz_dir_test.mkdir(parents=True, exist_ok=True)

    for i in range(df.shape[0]):
        data = df.iloc[i]
        filename = data['filename']
        train_val_test = data['train_val_test']
        src_path = npz_dir_train / f'{filename}.npz'
        if train_val_test == 0:
            dst_path = npz_dir_train / f'{filename}.npz'
        elif train_val_test == 1:
            dst_path = npz_dir_val / f'{filename}.npz'
        elif train_val_test == 2:
            dst_path = npz_dir_test / f'{filename}.npz'            
        shutil.move(src_path, dst_path)


if __name__ == '__main__':
    main()