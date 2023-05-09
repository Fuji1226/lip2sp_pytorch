from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random 
import pyopenjtalk
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
from data_process.phoneme_encode import classes2index_tts, pp_symbols


recorded_dir = Path("~/dataset/lip/np_files/face_aligned_0_50_gray/train").expanduser()
synth_dir = Path("~/dataset/lip/np_files_synth_corpus/face_aligned_0_50_gray/train").expanduser()
recorded_dir = recorded_dir / "F01_kablab" / "mspec80"
synth_dir = synth_dir / "F01_kablab" / "mspec80"
recorded_path_list = list(recorded_dir.glob("*.npz"))
synth_path_list = list(synth_dir.glob("*.npz"))
synth_path_list = random.sample(synth_path_list, int(len(synth_path_list) * 0.1))
all_path_list = recorded_path_list + synth_path_list
all_path_list = all_path_list[-1000:]

text_dir = Path("~/dataset/lip/utt").expanduser()


def get_recorded_synth_label(path):
    if ("ATR" in path.stem) or ("balanced" in path.stem):
        label = 1
    elif "BASIC5000" in path.stem:
        if int(str(path.stem).split("_")[1]) > 2500:
            label = 0
        else:
            label = 1
    else:
        label = 0
    return label


def get_utt_label(data_path):
    print("--- get utterance ---")
    path_text_label_list = []
    for path in tqdm(data_path):
        text_path = text_dir / f"{path.stem}.txt"
        df = pd.read_csv(str(text_path), header=None)
        text = df[0].values[0]
        label = get_recorded_synth_label(path)
        path_text_label_list.append([path, text, label])

    return path_text_label_list


path_text_label_list = get_utt_label(all_path_list)
data_path, text, label = path_text_label_list[0]
class_to_id, id_to_class = classes2index_tts()
text = pyopenjtalk.extract_fullcontext(text)
text = pp_symbols(text)
text = [class_to_id[t] for t in text]
print(text)