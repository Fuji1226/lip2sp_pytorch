from pathlib import Path
from tqdm import tqdm
import numpy as np


def main():
    lip_dir = Path("~/dataset/lip/np_files/lip_cropped_0.8_50_gray").expanduser()
    face_dir = Path("~/dataset/lip/np_files/face_aligned_0_50_gray").expanduser()
    save_dir = Path("~/dataset/lip/np_files/lip_face_merge").expanduser()
    speaker_list = ["F01_kablab"]
    data_split_list = ["train", "val", "test"]
    for speaker in speaker_list:
        for data_split in data_split_list:
            lip_dir_each = lip_dir / data_split / speaker / "mspec80"
            face_dir_each = face_dir / data_split / speaker / "mspec80"
            lip_path_list = list(lip_dir_each.glob("*.npz"))
            
            for lip_path in tqdm(lip_path_list):
                face_path = face_dir_each / f"{lip_path.stem}.npz"
                face_npz_key = np.load(str(face_path))
                lip_npz_key = np.load(str(lip_path))
                wav = face_npz_key["wav"]
                feature = face_npz_key["feature"]
                feat_add = face_npz_key["feat_add"]
                face = face_npz_key["lip"]
                lip = lip_npz_key["lip"]  
                save_dir_each = save_dir / data_split / speaker / "mspec80"
                save_dir_each.mkdir(parents=True, exist_ok=True)
                np.savez(
                    str(save_dir_each / lip_path.stem),
                    wav=wav,
                    lip=lip,
                    face=face,
                    feature=feature,
                    feat_add=feat_add,
                )
                
    
    
if __name__ == "__main__":
    main()