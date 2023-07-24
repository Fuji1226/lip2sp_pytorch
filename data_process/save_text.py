from pathlib import Path
import pandas as pd


def main():
    data_dir = Path('~/lip2sp_pytorch/csv').expanduser()
    save_dir = Path('~/dataset/lip/utt_small').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path_list = list(data_dir.glob('*'))

    for file_path in file_path_list:
        if file_path.stem == 'ATR503' or file_path.stem == 'balanced':
            df = pd.read_csv(str(file_path))
            for i in range(df.shape[0]):
                data = df.iloc[i]
                utt_num = data['utt_num']
                text = data['text']
                with open(str(save_dir / f'{file_path.stem}_{utt_num}.txt'), 'w') as f:
                    f.write(text)

        elif file_path.stem == 'BASIC5000':
            df = pd.read_csv(str(file_path))
            for i in range(df.shape[0]):
                data = df.iloc[i]
                utt_num = data['num']
                text = data['text']
                with open(str(save_dir / f'{file_path.stem}_{utt_num:04}.txt'), 'w') as f:
                    f.write(text)


if __name__ == '__main__':
    main()