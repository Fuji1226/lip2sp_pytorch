from pathlib import Path
from bs4 import BeautifulSoup


def main():
    data_dir = Path('/home/minami/aozorabunko/cards/000168/files')
    html_file_path_list = list(data_dir.glob('**/*.html'))
    for html_file_path in html_file_path_list:
        with open(str(html_file_path), 'r', encoding='cp932') as f:
            soup = BeautifulSoup(f, 'html.parser')
        author_elem = soup.find('h2', class_='author')
        if author_elem:
            print(author_elem.text)
    
    
if __name__ == "__main__":
    main()