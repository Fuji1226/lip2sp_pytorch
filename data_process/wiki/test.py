from pathlib import Path
import xml.etree.ElementTree as et
import unicodedata
import re
from tqdm import tqdm
import pyopenjtalk
import MeCab
import pandas as pd
import demoji
import neologdn
import random


class CustomMeCabTagger(MeCab.Tagger):

    COLUMNS = ['hyousou', 'hinnshi', 'hinnshi_sai1', 'hinnshi_sai2', 'hinnshi_sai3', 'katuyou_kata', 'katuyou_kei', 'gennkeki', 'yomi', 'hatuonn']

    def parseToDataFrame(self, text: str) -> pd.DataFrame:
        """テキストを parse した結果を Pandas DataFrame として返す"""
        results = []
        for line in self.parse(text).split('\n'):
            if line == 'EOS':
                break
            surface, feature = line.split('\t')
            feature = [None if f == '*' else f for f in feature.split(',')]
            results.append([surface, *feature])
        return pd.DataFrame(results, columns=type(self).COLUMNS)
    
    
def main():
    data_dir = Path('/home/minami/wiki_ja/text_fixed')
    data_path_list = list(data_dir.glob('**/wiki_*'))
    mecab = CustomMeCabTagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    
    for data_path in tqdm(data_path_list):
        tree = et.parse(str(data_path))
        root = tree.getroot()
        doc_list = root.findall('doc')
        for doc in doc_list:
            text = doc.text
            text = text.replace('\n', '').replace('\r', '')
            text = re.sub(r'[“”]', '', text)
            text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
            text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
            text = demoji.replace(string=text, repl='')
            text = re.sub(r'[!”#\$%&\’()*+,\-.\/:;?@[\\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠、。,？！｀＋￥％※→←↑↓△▽▷◁▲▼▶◀ゝ…☆]*', '', text)
            text = re.sub('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]', '', text)
            text = neologdn.normalize(text)
            text = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', text)
            n = random.randint(0, 9)
            text = re.sub(f'[0-9]+', str(n), text)
            text = text.lower()
            
            if len(text) < 30:
                continue
            if re.search(f'[a-z]+', text) is not None:
                continue
            
            result = mecab.parseToDataFrame(text)
            result = result.loc[~result['yomi'].isna()]
            result['phoneme'] = result['yomi'].apply(lambda x: pyopenjtalk.g2p(x, join=False))
            result = result.loc[~(result['phoneme'] == '')]
            phoneme = []
            for p in result['phoneme'].values:
                phoneme += p
                    
                    
    
if __name__ == '__main__':
    main()