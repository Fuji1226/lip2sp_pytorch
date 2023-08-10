from pathlib import Path


def get_file_size(file_path):
    if file_path.is_file():
        return file_path.stat().st_size
    else:
        return None


def main():
    data_dir = Path('~/Lip2Wav/Dataset_split_wide_fps25').expanduser()
    speaker_list = ['chem', 'chess', 'dl', 'hs', 'eh']
    for speaker in speaker_list:
        print(f'speaker = {speaker}')
        data_dir_spk = data_dir / speaker / 'intervals'
        data_path_list = list(data_dir_spk.glob('*/*.mp4'))
        error_list = []
        
        for data_path in data_path_list:
            size = get_file_size(data_path)
            if size == 0:
                error_list.append(data_path)
        
        print(f'n_data = {len(data_path_list)}, n_data_error = {len(error_list)}')
        print()


if __name__ == '__main__':
    main()