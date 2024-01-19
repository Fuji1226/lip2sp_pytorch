from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    data_dir = Path('~/wiki_ja/csv').expanduser()
    data_path_list = list(data_dir.glob('**/*.csv'))
    train_data_path_list, val_test_data_path_list = train_test_split(
        data_path_list,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )
    val_data_path_list, test_data_path_list = train_test_split(
        val_test_data_path_list,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )
    df_train = pd.DataFrame(
        {
            'data_path': train_data_path_list,
            'data_split': 'train',
        }
    )
    df_val = pd.DataFrame(
        {
            'data_path': val_data_path_list,
            'data_split': 'val',
        }
    )
    df_test = pd.DataFrame(
        {
            'data_path': test_data_path_list,
            'data_split': 'test',
        }
    )
    df = pd.concat([df_train, df_val, df_test], axis=0)
    save_path = Path('~/wiki_ja/data_split.csv').expanduser()
    df.to_csv(str(save_path), index=False)
    
    
if __name__ == '__main__':
    main()