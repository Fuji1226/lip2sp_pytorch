from pathlib import Path
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def text_cleanse_df(zip_file_path):
    try:
        df = pd.read_csv(str(zip_file_path), encoding='cp932', names=['text'])
    except pd.errors.ParserError as e:
        print(f'{e}')
        print(f'Skip {zip_file_path}')
        return None
    except ValueError as e:
        print(f'{e}')
        print(f'Skip {zip_file_path}')
        return None
    
    # 本文の先頭を探す（'---…'区切りの直後から本文が始まる前提）
    head_tx = list(df[df['text'].str.contains(
        '-------------------------------------------------------')].index)
    # 本文の末尾を探す（'底本：'の直前に本文が終わる前提）
    atx = list(df[df['text'].str.contains('底本：')].index)
    if head_tx == []:
        author_name = None
        html_file_path_list = list(zip_file_path.parent.glob('**/*.html'))
        for html_file_path in html_file_path_list:
            with open(str(html_file_path), 'r', encoding='cp932') as f:
                soup = BeautifulSoup(f, 'html.parser')
            author_elem = soup.find('h2', class_='author')
            if author_elem:
                author_name = author_elem.text
                break
        if author_name is None:
            print('Author name was not found.')
            print(f'Skip {zip_file_path}')
            return None
        
        # もし'---…'区切りが無い場合は、作家名の直後に本文が始まる前提
        head_tx = list(df[df['text'].str.contains(author_name)].index)
        head_tx_num = head_tx[0]+1
    else:
        # 2個目の'---…'区切り直後から本文が始まる
        head_tx_num = head_tx[1]+1

    try:
        df_e = df.iloc[head_tx_num:atx[0]]
    except IndexError as e:
        print(f'{e}')
        print(f'Skip {zip_file_path}')
        return None

    # 青空文庫の書式削除
    df_e = df_e.replace({'text': {'《.*?》': ''}}, regex=True)
    df_e = df_e.replace({'text': {'［.*?］': ''}}, regex=True)
    df_e = df_e.replace({'text': {'｜': ''}}, regex=True)

    # 字下げ（行頭の全角スペース）を削除
    df_e = df_e.replace({'text': {'　': ''}}, regex=True)

    # 節区切りを削除
    df_e = df_e.replace({'text': {'^.$': ''}}, regex=True)
    df_e = df_e.replace({'text': {'^―――.*$': ''}}, regex=True)
    df_e = df_e.replace({'text': {'^＊＊＊.*$': ''}}, regex=True)
    df_e = df_e.replace({'text': {'^×××.*$': ''}}, regex=True)

    # 記号、および記号削除によって残ったカッコを削除
    df_e = df_e.replace({'text': {'―': ''}}, regex=True)
    df_e = df_e.replace({'text': {'…': ''}}, regex=True)
    df_e = df_e.replace({'text': {'※': ''}}, regex=True)
    df_e = df_e.replace({'text': {'「」': ''}}, regex=True)

    # 一文字以下で構成されている行を削除
    df_e['length'] = df_e['text'].map(lambda x: len(x))
    df_e = df_e[df_e['length'] > 1]

    # インデックスがずれるので振りなおす
    df_e = df_e.reset_index().drop(['index'], axis=1)

    # 空白行を削除する（念のため）
    df_e = df_e[~(df_e['text'] == '')]

    # インデックスがずれるので振り直し、文字の長さの列を削除する
    df_e = df_e.reset_index().drop(['index', 'length'], axis=1)
    
    df_e['title'] = df['text'][0]
    return df_e


def main():
    data_dir = Path('/home/minami/aozorabunko/cards')
    zip_file_path_list = list(data_dir.glob('**/*.zip'))
    for zip_file_path in tqdm(zip_file_path_list):
        df_cleaned = text_cleanse_df(zip_file_path)
        if df_cleaned is None:
            print('dataframe could not be loaded.')
    
    
if __name__ == "__main__":
    main()