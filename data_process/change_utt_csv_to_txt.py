from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os


text_dir = Path("~/dataset/lip/utt").expanduser()
data_path = list(text_dir.glob("*.txt"))

# for path in tqdm(data_path):
#     df = pd.read_csv(str(path))
#     text = df["text"].values[0]
#     with open(str(text_dir / f"{path.stem}.txt"), "w") as f:
#         f.write(text)


# for path in tqdm(data_path):
#     os.remove(str(path))