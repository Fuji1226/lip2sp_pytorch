from pathlib import Path
import pandas as pd
import random
import shutil
from tqdm import tqdm


def make_fixed_csv():
    data_split_csv_dir = Path('~/dataset/lip/data_split_csv').expanduser()
    train_df = pd.read_csv(str(data_split_csv_dir / 'train.csv'), header=None)
    val_df = pd.read_csv(str(data_split_csv_dir / 'val.csv'), header=None)
    test_df = pd.read_csv(str(data_split_csv_dir / 'test.csv'), header=None)
    filename_list = train_df[0].to_list() + val_df[0].to_list() + test_df[0].to_list()
    
    filename_list_atr = []
    filename_list_basic = []
    filename_list_balanced = []
    for filename in filename_list:
        if 'ATR503' in filename:
            filename_list_atr.append(filename)
        elif 'BASIC5000' in filename:
            filename_list_basic.append(filename)
        elif 'balanced' in filename:
            filename_list_balanced.append(filename)

    filename_list_train = []
    filename_list_val = []
    filename_list_test = []
    for filename in filename_list_atr:
        if 'i' in filename:
            filename_list_val.append(filename)
        elif 'j' in filename:
            filename_list_test.append(filename)
        else:
            filename_list_train.append(filename)

    filename_list_basic = random.sample(filename_list_basic, len(filename_list_basic))
    filename_list_balanced = random.sample(filename_list_balanced, len(filename_list_balanced))
    filename_list_train += filename_list_basic[:int(len(filename_list_basic) * 0.95)]
    filename_list_val += filename_list_basic[int(len(filename_list_basic) * 0.95):]
    filename_list_train += filename_list_balanced[:int(len(filename_list_balanced) * 0.95)]
    filename_list_val += filename_list_balanced[int(len(filename_list_balanced) * 0.95):]

    train_df = pd.DataFrame({'filename': filename_list_train})
    val_df = pd.DataFrame({'filename': filename_list_val})
    test_df = pd.DataFrame({'filename': filename_list_test})

    train_df = train_df.sort_values(['filename']).reset_index(drop=True)
    val_df = val_df.sort_values(['filename']).reset_index(drop=True)
    test_df = test_df.sort_values(['filename']).reset_index(drop=True)

    save_dir = Path('~/dataset/lip/data_split_csv_fix').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(str(save_dir / 'train.csv'), index=False)
    val_df.to_csv(str(save_dir / 'val.csv'), index=False)
    test_df.to_csv(str(save_dir / 'test.csv'), index=False)


def search_file_and_move(df, dst_data, speaker_list, data_list, data_dir, debug):
    for filename in tqdm(df['filename'].to_list()):
        for speaker in speaker_list:
            for data in data_list:
                src_path = data_dir / data / speaker / 'mspec80' / f'{filename}.npz'
                if src_path.exists():
                    if data == dst_data:
                        continue
                    else:
                        dst_path = Path(str(src_path).replace(data, dst_data))
                        shutil.move(src=str(src_path), dst=str(dst_path))


def main():
    data_dir = Path('~/dataset/lip/data_split_csv_fix').expanduser()
    train_df = pd.read_csv(str(data_dir / 'train.csv'))
    val_df = pd.read_csv(str(data_dir / 'val.csv'))
    test_df = pd.read_csv(str(data_dir / 'test.csv'))

    debug = False
    
    data_dir = Path('~/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray').expanduser()
    speaker_list = ['F01_kablab', 'F02_kablab', 'M01_kablab', 'M04_kablab']
    data_list = ['train', 'val', 'test']

    search_file_and_move(train_df, 'train', speaker_list, data_list, data_dir, debug)
    search_file_and_move(val_df, 'val', speaker_list, data_list, data_dir, debug)
    search_file_and_move(test_df, 'test', speaker_list, data_list, data_dir, debug)
    


if __name__ == '__main__':
    main()