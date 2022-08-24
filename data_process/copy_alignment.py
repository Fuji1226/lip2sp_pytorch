"""
juliusを用いて音素アラインメントを行った時にミスしているデータがたまにあるので,それを取り除いて使えるデータをコピーする
"""

import os
from pathlib import Path
import shutil
from tqdm import tqdm


def get_alignment(data_root):
    """
    音素アラインメントを行った.labファイルまでのパスを取得
    """
    data_path = list(data_root.glob("*.lab"))
    assert data_path is not None
    return data_path


def check_alignment(data_path):
    """
    アラインメントをミスしている時があるので,そのパスを取り除く
    """
    exist_data_path_train = []
    exist_data_path_test = []


    for p in tqdm(data_path, total=len(data_path)):
        with open(p, "r") as f:
            data = f.read()

        data = data.replace("\n", " ")
        data = data.split(" ")

        data = data[2::3]
        
        if data != []:
            if "_j" in Path(p).stem:
                exist_data_path_test.append(p)
            else:
                exist_data_path_train.append(p)
        else:
            print(p)

    return exist_data_path_train, exist_data_path_test


def main():
    data_root = Path("~/dataset/segmentation-kit/wav").expanduser()
    data_path = get_alignment(data_root)

    # たまにミスしているデータがあるので，それを除く
    exist_data_path_train, exist_data_path_test = check_alignment(data_path)

    save_dir = Path("~/dataset/lip/np_files/lip_cropped_9696_time_only").expanduser()
    save_dir_train = save_dir / "train" / "F01_kablab"
    save_dir_test = save_dir / "test" / "F01_kablab"
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    for p in tqdm(exist_data_path_train, total=len(exist_data_path_train)):
        shutil.copy(str(p), save_dir_train)

    for p in tqdm(exist_data_path_test, total=len(exist_data_path_test)):
        shutil.copy(str(p), save_dir_test)

    
if __name__ == "__main__":
    main()