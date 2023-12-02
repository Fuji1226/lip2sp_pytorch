from pathlib import Path
import pandas as pd


def main():
    kablab_df_path = Path('~/dataset/lip/data_split_csv/kablab.csv').expanduser()
    kablab_df = pd.read_csv(str(kablab_df_path))
    # filename, data_split, speaker, corpus
    
    data_dir = Path('~/tcd_timit').expanduser()
    data_path_list = list(data_dir.glob('**/*.mp4'))
    filename_list = []
    speaker_list = []
    for data_path in data_path_list:
        filename = data_path.stem
        speaker = data_path.parents[1].name
        filename_list.append(filename)
        speaker_list.append(speaker)
    timit_df = pd.DataFrame(
        {
            'filename': filename_list,
            'speaker': speaker_list,
        }
    )
    timit_df = timit_df.groupby('speaker').apply(lambda x: x.sample(frac=1, random_state=42)).reset_index(drop=True)
    timit_df['flag'] = timit_df.groupby('speaker').cumcount()
    flag_max = timit_df.groupby('speaker')['flag'].max()
    flag_max.name = 'flag_max'
    timit_df = timit_df.merge(flag_max, on='speaker', how='left')
    timit_df['flag'] = (timit_df['flag']) / timit_df['flag_max']
    timit_df.loc[timit_df['flag'] <= 0.9, 'data_split'] = 'train'
    timit_df.loc[(timit_df['flag'] > 0.9) & (timit_df['flag'] <= 0.95), 'data_split'] = 'val'
    timit_df.loc[(timit_df['flag'] > 0.95), 'data_split'] = 'test'
    
    save_path = Path('~/dataset/lip/data_split_csv/tcd_timit.csv').expanduser()
    timit_df.to_csv(str(save_path), index=False)
    



if __name__ == '__main__':
    main()