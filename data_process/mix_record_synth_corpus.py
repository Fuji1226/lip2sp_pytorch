import shutil
from pathlib import Path
from tqdm import tqdm

data_dir_recorded = Path(f"~/dataset/lip/np_files/face_aligned_0_50_gray").expanduser()
data_dir_synth = Path(f"~/dataset/lip/np_files_synth_corpus/face_aligned_0_50_gray").expanduser()
save_dir = Path(f"~/dataset/lip/np_files_recorded_and_synth/face_aligned_0_50_gray").expanduser()
data_list = ["train", "val", "test"]
speaker_list = ["F01_kablab"]
name = "mspec80"


def copy_data(data_dir):
    for speaker in speaker_list:
        for data in data_list:
            print(speaker, data)
            data_dir_spk = data_dir / data / speaker / name
            data_path_spk = list(data_dir_spk.glob("*.npz"))
            save_dir_spk = save_dir / data / speaker / name
            save_dir_spk.mkdir(parents=True, exist_ok=True)
            print(len(data_path_spk))
            print(save_dir_spk)
            for path in tqdm(data_path_spk):
                shutil.copy(str(path), str(save_dir_spk))    

def main():
    print("copy recorded data")
    copy_data(data_dir_recorded)
    print("copy synthesized data")
    copy_data(data_dir_synth)
                        
            
if __name__ == "__main__":
    main()