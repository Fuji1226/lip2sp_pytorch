from pathlib import Path
from tqdm import tqdm


data_dir = Path("~/lrw/lipread_mp4").expanduser()
word_dir_list = list(data_dir.glob("*"))
data_list = ["train", "val", "test"]
n_train = 0
n_val = 0
n_test = 0

for word_dir in tqdm(word_dir_list):
    for data in data_list:
        data_dir = word_dir / data
        data_path_list = list(data_dir.glob("*.mp4"))
        if data == "train":
            n_train += len(data_path_list)
        elif data == "val":
            n_val += len(data_path_list)
        elif data == "test":
            n_test += len(data_path_list)
            
print(n_train, n_val, n_test)