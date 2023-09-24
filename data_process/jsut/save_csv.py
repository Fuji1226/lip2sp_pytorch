from pathlib import Path
import pandas as pd
import numpy as np

# csvファイルに学習・検証・テストを記す
def main():
    np.random.seed(42)
    df = pd.read_csv('/home/minami/dataset/jsut_ver1.1/filename.csv')
    df = df[['corpus', 'filename']]
    corpus_list = df['corpus'].unique()
    for corpus in corpus_list:
        num_samples = df.loc[df['corpus'] == corpus, 'filename'].nunique()
        train_ratio = 0.9
        val_ratio = 0.05
        test_ratio = 0.05
        split_index = np.random.choice([0, 1, 2], num_samples, p=[train_ratio, val_ratio, test_ratio])
        df.loc[df['corpus'] == corpus, 'train_val_test'] = split_index

    df.to_csv('/home/minami/dataset/jsut_ver1.1/filename.csv', index=False)
    


if __name__ == '__main__':
    main()