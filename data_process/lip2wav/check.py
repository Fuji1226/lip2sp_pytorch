from pathlib import Path
import os
import numpy as np
import re
import av


def main():
    data_dir = Path("~/Lip2Wav/Dataset_fps25").expanduser()
    speaker_list = ["chem", "chess", "dl", "eh", "hs"]
    for speaker in speaker_list:
        data_dir_spk = data_dir / speaker
        data_path_list = sorted(list(data_dir_spk.glob("*/*/*.mp4")))
        
        for data_path in data_path_list:
            container = av.open(str(data_path))
            stream = container.streams.video[0]
            fps = int(stream.rate)
            print(fps)
    
    
if __name__ == "__main__":
    main()