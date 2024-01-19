from pathlib import Path
import re
from tqdm import tqdm
import pyopenjtalk
import MeCab
import pandas as pd
import demoji
import neologdn
import xml.etree.ElementTree as et
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
from data_process.text_process_utils import CustomMeCabTagger


def main():
    mecab = CustomMeCabTagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    data_dir = Path('/home/minami/wiki_ja/text_fixed')
    data_path_list = list(data_dir.glob('**/wiki_*'))
    
    for data_path in tqdm(data_path_list):
        tree = et.parse(str(data_path))
        root = tree.getroot()
        doc_list = root.findall('doc')
        for idx, doc in enumerate(doc_list):
            try:
                text = doc.text
                text = text.replace('\n', '').replace('\r', '')
                text = re.sub(r'[“”]', '', text)
                text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
                text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
                text = demoji.replace(string=text, repl='')
                text = re.sub(r'[!”#\$%&\’()*+,\-.\/:;?@[\\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠,？！｀＋￥％※→←↑↓△▽▷◁▲▼▶◀ゝ…☆]*', '', text)
                text = neologdn.normalize(text)
                text = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', text)
                text = re.sub(f'[0-9]+', '0', text)
                if len(text) < 50:
                    continue
                if re.search(f'[a-zA-Z]+', text) is not None:
                    continue
                text_parsed = mecab.parseToDataFrame(text)
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
                
                save_path = Path(str(data_path).replace('text_fixed', 'csv')) / f'{idx}.csv'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                text_parsed.to_csv(str(save_path), index=False)
            except:
                continue
            
    
if __name__ == '__main__':
    main()