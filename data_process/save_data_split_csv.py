from pathlib import Path
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
random.seed(42)


def data_split_kablab():
    audio_dir = Path('~/dataset/lip/wav').expanduser()
    speaker_list = list(audio_dir.glob('*'))
    speaker_list = [s.stem for s in speaker_list]
    train_ratio = 0.95
    df_list = []

    for speaker in speaker_list:
        audio_path_list = list((audio_dir / speaker).glob('*.wav'))
        filename_list = [f.stem for f in audio_path_list]
        
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
        filename_list_train += filename_list_basic[:int(len(filename_list_basic) * train_ratio)]
        filename_list_val += filename_list_basic[int(len(filename_list_basic) * train_ratio):]
        filename_list_train += filename_list_balanced[:int(len(filename_list_balanced) * train_ratio)]
        filename_list_val += filename_list_balanced[int(len(filename_list_balanced) * train_ratio):]

        train_df = pd.DataFrame({'filename': filename_list_train, 'data_split': 'train'})
        val_df = pd.DataFrame({'filename': filename_list_val, 'data_split': 'val'})
        test_df = pd.DataFrame({'filename': filename_list_test, 'data_split': 'test'})
        df = pd.concat([train_df, val_df, test_df])
        df['speaker'] = speaker
        df.loc[df['filename'].str.contains('ATR'), 'corpus'] = 'ATR'
        df.loc[df['filename'].str.contains('BASIC5000'), 'corpus'] = 'BASIC5000'
        df.loc[df['filename'].str.contains('balanced'), 'corpus'] = 'balanced'
        df_list.append(df)

    df = pd.concat(df_list)
    return df


def data_split_hifi_captain():
    data_dir = Path('~/dataset/hi-fi-captain/ja-JP').expanduser()
    speaker_list = ['female', 'male']
    data_list = ['train_parallel', 'train_non_parallel', 'dev', 'eval']
    df_list = []

    for speaker in speaker_list:
        for data in data_list:
            data_dir_spk = data_dir / speaker / 'wav' / data
            data_path_list = data_dir_spk.glob('*.wav')
            filename_list = [f.stem for f in data_path_list]
            df = pd.DataFrame({"filename": filename_list})
            if data == 'train_parallel' or data == 'train_non_parallel':
                df['data_split'] = 'train'
            elif data == 'dev':
                df['data_split'] = 'val'
            elif data == 'eval':
                df['data_split'] = 'test'
            df['speaker'] = speaker
            df['parent_dir'] = data
            df_list.append(df)

    df = pd.concat(df_list)
    return df


def data_split_jsut():
    data_dir = Path('~/dataset/jsut_ver1.1').expanduser()
    data_list = data_dir.glob('*')
    df_list = []

    for data in data_list:
        if not data.is_dir():
            continue
        data_path_list = data.glob('**/*.wav')
        filename_list = [f.stem for f in data_path_list]
        filename_list = random.sample(filename_list, len(filename_list))
        total_length = len(filename_list)
        train_length = int(total_length * 0.9)
        val_length = int(total_length * 0.05)
        filename_list_train = filename_list[:train_length]
        filename_list_val = filename_list[train_length:train_length + val_length]
        filename_list_test = filename_list[train_length + val_length:]
        train_df = pd.DataFrame({"filename": filename_list_train, 'data_split': 'train'})
        val_df = pd.DataFrame({"filename": filename_list_val, 'data_split': 'val'})
        test_df = pd.DataFrame({"filename": filename_list_test, 'data_split': 'test'})
        df = pd.concat([train_df, val_df, test_df])
        df_list.append(df)
    
    df = pd.concat(df_list)
    return df


def data_split_jvs():
    data_dir = Path('~/dataset/jvs_ver1').expanduser()
    data_path_list = data_dir.glob('**/*.wav')
    speaker_list = []
    data_list = []
    filename_list = []
    data_split_list = []

    for data_path in data_path_list:
        speaker = data_path.parents[2].name
        data = data_path.parents[1].name
        filename = data_path.stem
        speaker_num = int(str(speaker).replace('jvs', ''))
        if speaker_num > 90:
            data_split = 'test'
        elif speaker_num > 80:
            data_split = 'val'
        else:
            data_split = 'train'

        speaker_list.append(speaker)
        data_list.append(data)
        filename_list.append(filename)
        data_split_list.append(data_split)
        
    df = pd.DataFrame(
        {
            'speaker': speaker_list,
            'data': data_list,
            'filename': filename_list,
            'data_split': data_split_list,
        }
    )
    return df


def data_split_vctk():
    data_dir = Path('~/VCTK-Corpus').expanduser()
    df_info = pd.read_csv(str(data_dir / 'speaker-info.txt'))
    data_path_list = data_dir.glob('**/*.wav')
    speaker_list = []
    filename_list = []
    
    for data_path in data_path_list:
        speaker = data_path.parents[0].name
        filename = data_path.stem
        speaker_list.append(speaker)
        filename_list.append(filename)

    df = pd.DataFrame(
        {
            'speaker': speaker_list,
            'filename': filename_list,
        }
    )
    speaker_list_train, speaker_list_test = train_test_split(df['speaker'].unique(), test_size=0.2)
    speaker_list_test, speaker_list_val = train_test_split(speaker_list_test, test_size=0.5)
    df.loc[df['speaker'].isin(speaker_list_train), 'data_split'] = 'train'
    df.loc[df['speaker'].isin(speaker_list_val), 'data_split'] = 'val'
    df.loc[df['speaker'].isin(speaker_list_test), 'data_split'] = 'test'
    df['ID'] = df['speaker'].str.replace('p', '').astype(int)
    df = df.merge(df_info, on='ID', how='left')
    return df
    

def main():
    save_dir = Path('~/dataset/lip/data_split_csv').expanduser()
    # df = data_split_kablab()
    # df.to_csv(str(save_dir / 'kablab.csv'), index=False)
    # df = data_split_hifi_captain()
    # df.to_csv(str(save_dir / 'hifi_captain.csv'), index=False)
    # df = data_split_jsut()
    # df.to_csv(str(save_dir / 'jsut.csv'), index=False)
    # df = data_split_jvs()
    # df.to_csv(str(save_dir / 'jvs.csv'), index=False)\
    df = data_split_vctk()
    df.to_csv(str(save_dir / 'vctk.csv'), index=False)


if __name__ == '__main__':
    main()