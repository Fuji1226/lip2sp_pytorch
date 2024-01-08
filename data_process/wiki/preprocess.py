import subprocess
from pathlib import Path
from tqdm import tqdm


def main():
    data_dir = Path('/home/minami/wiki_ja/text')
    data_path_list = list(data_dir.glob('**/wiki_*'))
    for data_path in tqdm(data_path_list):
        save_path = Path(str(data_path).replace('text', 'text_fixed'))
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        with open(str(data_path), 'r') as f:
            lines = f.readlines()
        lines.insert(0, '<xml>\n')
        lines.append('</xml>\n')
        with open(str(save_path), 'w') as f:
            f.writelines(lines)
    
    
if __name__ == "__main__":
    main()