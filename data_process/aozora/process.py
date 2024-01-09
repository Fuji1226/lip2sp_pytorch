from pathlib import Path
import re
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm


def main():
    txt_data_dir = Path('/home/minami/aozorabunko_text/cards')
    html_data_dir = Path('/home/minami/aozorabunko/cards')
    txt_file_path_list = list(txt_data_dir.glob('**/*.txt'))
    skip_list = []
    cnt_dict_head = defaultdict(int)
    path_dict_head = defaultdict(list)
    cnt_dict_author = defaultdict(int)
    path_dict_author = defaultdict(list)
    cnt_dict_end = defaultdict(int)
    path_dict_end = defaultdict(list)
    result_list = []
    
    for txt_file_path in tqdm(txt_file_path_list):
        try:
            with open(str(txt_file_path), 'r', encoding='cp932') as f:
                line_list = f.readlines()
        except UnicodeDecodeError as e:
            skip_list.append(txt_file_path)
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
            print(txt_file_path)
        
        # line_list = [line.strip() for line in    line_list]
        # head_list = []
        # author_list = []
        # end_list = []
        
        # for i, line in enumerate(line_list):
        #     if re.match('^-+$', line):
        #         head_list.append((i, line))
        #     elif re.match(author_name, line):
        #         author_list.append((i, author_name))
        #     elif re.match('^底本：', line):
        #         end_list.append((i, line))
                
        # cnt_dict_head[len(head_list)] += 1
        # path_dict_head[len(head_list)].append(txt_file_path)
        # cnt_dict_author[len(author_list)] += 1
        # path_dict_author[len(author_list)].append(txt_file_path)
        # cnt_dict_end[len(end_list)] += 1
        # path_dict_end[len(end_list)].append(txt_file_path)
        
    
    
if __name__ == "__main__":
    main()