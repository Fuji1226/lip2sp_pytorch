"""
detを使用して顔全体を切り取る
"""

import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import random


debug = True
debug_iter = 10

margin = 0.3
speaker = "F01_kablab"
data_root = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
det_dir = Path(f"~/dataset/lip/det_debug/{speaker}").expanduser()

if debug:
    save_dir = Path(f"~/dataset/lip/face_cropped_debug/{speaker}").expanduser()    
else:
    save_dir = Path(f"~/dataset/lip/face_cropped/{speaker}").expanduser()


def get_crop_info(det_path):
    df = pd.read_csv(str(det_path), header=None)

    coords_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        coords_list.append(coords)

    coords_mean = []
    crop_size = 0
    for coords in coords_list:
        coords_mean.append([np.mean(coords[:2]).astype("int"), np.mean(coords[2:]).astype("int")])
        width = coords[1] - coords[0]
        height = coords[3] - coords[2]
        if width > height:
            each_crop_size = width
        else:
            each_crop_size = height
        
        if each_crop_size > crop_size:
            crop_size = each_crop_size

    print(coords_mean)
    print(crop_size)

    return coords_mean, crop_size

def face_crop(data_path, det_path, save_dir):
    coords_mean, crop_size = get_crop_info(det_path)
    movie = cv2.VideoCapture(str(data_path))
    fps = movie.get(cv2.CAP_PROP_FPS)
    n_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)

    assert n_frame == len(coords_mean)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(f"{save_dir}/{data_path.stem}_crop.mp4", int(fourcc), fps, (int(crop_size), int(crop_size)))

    iter_cnt = 0
    while True:
        ret, frame = movie.read()
        if ret == False:
            break
        
        mouth_area = [
            int(coords_mean[iter_cnt][1] - crop_size//2),   # 下端
            int(coords_mean[iter_cnt][1] + crop_size//2),   # 上端
            int(coords_mean[iter_cnt][0] - crop_size//2),   # 左端
            int(coords_mean[iter_cnt][0] + crop_size//2)    # 右端
        ]
        iter_cnt += 1

        out.write(frame[mouth_area[0]:mouth_area[1], mouth_area[2]:mouth_area[3]])

    out.release()


def main():
    os.makedirs(save_dir, exist_ok=True)
    data_path_list = sorted(list(data_root.glob("*.mp4")))
    if debug:
        data_path_list = random.sample(data_path_list, len(data_path_list))

    print(f"speaker = {speaker}, margin = {margin}")

    iter_cnt = 0

    for data_path in tqdm(data_path_list):
        det_path = det_dir / f"{data_path.stem}_det.csv"

        if det_path.exists():
            face_crop(data_path, det_path, save_dir)
        else:
            continue

        iter_cnt += 1
        if debug:
            if debug_iter < iter_cnt:
                break


if __name__ == "__main__":
    main()
