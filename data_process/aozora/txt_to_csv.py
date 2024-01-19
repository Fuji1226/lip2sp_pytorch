from pathlib import Path
import re
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm
import neologdn
import demoji
import MeCab
import pyopenjtalk
import pandas as pd
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
from data_process.text_process_utils import CustomMeCabTagger


def main():
    mecab = CustomMeCabTagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    txt_data_dir = Path('/home/minami/aozorabunko_text/cards')
    html_data_dir = Path('/home/minami/aozorabunko/cards')
    txt_file_path_list = list(txt_data_dir.glob('**/*.txt'))
    
    for txt_file_path in tqdm(txt_file_path_list):
        try:
            with open(str(txt_file_path), 'r', encoding='cp932') as f:
                line_list = f.readlines()
        except UnicodeDecodeError as e:
            continue
        
        author_name = None
        author_num = txt_file_path.parents[2].name
        html_file_path_list = list((html_data_dir / author_num / 'files').glob('**/*.html'))
        for html_file_path in html_file_path_list:
            with open(str(html_file_path), 'r', encoding='cp932') as f:
                soup = BeautifulSoup(f, 'html.parser')
            author_elem = soup.find('h2', class_='author')
            if author_elem:
                author_name = author_elem.text
                break
        
        if author_name is None:
            '''
            /home/minami/aozorabunko_text/cards/000869/files/2707_ruby_6189/2707_ruby_6189.txt
            /home/minami/aozorabunko_text/cards/000323/files/2171_ruby/2171_ruby.txt
            /home/minami/aozorabunko_text/cards/001790/files/56572_txt_51803/56572_txt_51803.txt
            /home/minami/aozorabunko_text/cards/001189/files/45257_ruby_18658/45257_ruby_18658.txt
            /home/minami/aozorabunko_text/cards/000149/files/802_txt/802_txt.txt
            /home/minami/aozorabunko_text/cards/001710/files/55214_txt_49096/55214_txt_49096.txt
            /home/minami/aozorabunko_text/cards/001710/files/55213_txt_49097/55213_txt_49097.txt
            /home/minami/aozorabunko_text/cards/001728/files/55346_txt_53428/55346_txt_53428.txt
            '''
            continue
        
        haihun_line_index = []
        finish_line_index = []
        for i, line in enumerate(line_list):
            if re.match('-------------------------------------------------------', line):
                haihun_line_index.append(i)
            elif re.match('^(底本：)', line):
                finish_line_index.append(i)
        try:
            line_list = line_list[haihun_line_index[-1]:finish_line_index[0]]
        except IndexError as e:
            continue
    
        for idx, line in enumerate(line_list):
            try:
                line = re.sub('《.*?》', '', line)
                line = re.sub('［.*?］', '', line)
                line = re.sub('｜', '', line)
                line = re.sub('　', '', line)
                line = re.sub('^―――.*$', '', line)
                line = re.sub('^＊＊＊.*$', '', line)
                line = re.sub('^×××.*$', '', line)
                line = re.sub('―', '', line)
                line = re.sub('…', '', line)
                line = re.sub('※', '', line)
                line = re.sub('「」', '', line)
                line = line.replace('\n', '').replace('\r', '')
                line = re.sub(r'[“”]', '', line)
                line = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', line)
                line = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', line)
                line = demoji.replace(string=line, repl='')
                line = re.sub(r'[!”#\$%&\’()*+,\-.\/:;?@[\\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠,？！｀＋￥％※→←↑↓△▽▷◁▲▼▶◀ゝ…☆]*', '', line)
                line = neologdn.normalize(line)
                line = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', line)
                line = re.sub(f'[0-9]+', '0', line)
                if len(line) < 50:
                    continue
                if re.search(f'[a-zA-Z]+', line) is not None:
                    continue
                text_parsed = mecab.parseToDataFrame(line)
                text_parsed = text_parsed.loc[~text_parsed['yomi'].isna()]
                text_parsed['phoneme'] = text_parsed['yomi'].apply(lambda x: pyopenjtalk.g2p(x, join=False))
                text_parsed = text_parsed.loc[~(text_parsed['phoneme'] == '')]
                text_parsed.loc[(text_parsed['hyousou'] == '。') | (text_parsed['hyousou'] == '、'), 'phoneme'] = 'pau'
                sentence_id_list = []
                sentence_id = 0
                for i, row in text_parsed.iterrows():
                    if row['hyousou'] == '。':
                        sentence_id_list.append(sentence_id)
                        sentence_id += 1
                    else:
                        sentence_id_list.append(sentence_id)
                text_parsed['sentence_id'] = sentence_id_list
                
                save_path = Path(str(txt_file_path).replace('cards', 'csv').replace('.txt', '')) / f'{idx}.csv'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                text_parsed.to_csv(str(save_path), index=False)
            except:
                continue
        
    
if __name__ == "__main__":
    main()