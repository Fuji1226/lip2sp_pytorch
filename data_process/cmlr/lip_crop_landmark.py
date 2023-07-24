"""
save_landmark.pyで保存したランドマークを使用して口唇部分を切り取る
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

margin = 0.8
data_dir = Path(f"~/cmlr").expanduser()
landmark_dir = data_dir / "landmark_debug"

if debug:
    save_dir = Path(f"~/cmlr/lip_cropped_debug_{margin}").expanduser()
else:
    save_dir = Path(f"~/cmlr/lip_cropped_{margin}").expanduser()


def get_crop_info(landmark_path):
    df = pd.read_csv(str(landmark_path), header=None)

    coords_list = []
    for i in range(len(df)):
        coords = df.iloc[i][96:].values
        xy_list = []
        for j in range(0, len(coords), 2):
            xy_list.append([coords[j], coords[j + 1]])
        
        coords_list.append(xy_list)

    coords_mean = []
    crop_size = 0
    for coords in coords_list:
        coords_mean.append(np.mean(coords, axis=0).astype("int"))

        left_point = coords[0][0]
        right_point = coords[6][0]    
        each_crop_size = right_point - left_point

        if each_crop_size > crop_size:
            crop_size = each_crop_size

    crop_size += int(crop_size * margin)
    return coords_mean, crop_size


def lip_crop(data_path, landmark_path, save_dir):
    coords_mean, crop_size = get_crop_info(landmark_path)
    movie = cv2.VideoCapture(str(data_path))
    fps = movie.get(cv2.CAP_PROP_FPS)
    n_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)

    assert n_frame == len(coords_mean)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save_path = save_dir / data_path.parents[1].name / data_path.parents[0].name
    os.makedirs(save_path, exist_ok=True)
    out = cv2.VideoWriter(f"{save_path}/{data_path.stem}_crop.mp4", int(fourcc), fps, (int(crop_size), int(crop_size)))

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

    data_path_list = []
    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mp4"):
                data_path_list.append(Path(os.path.join(curdir, file)))

    if debug:
        data_path_list = random.sample(data_path_list, len(data_path_list))

    iter_cnt = 0

    for data_path in tqdm(data_path_list):
        landmark_path = landmark_dir / data_path.parents[1].name / data_path.parents[0].name / f"{data_path.stem}.csv"

        if landmark_path.exists():
            lip_crop(data_path, landmark_path, save_dir)
        else:
            continue

        iter_cnt += 1
        if debug:
            if debug_iter < iter_cnt:
                break


if __name__ == "__main__":
    main()
