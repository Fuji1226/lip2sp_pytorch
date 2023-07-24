import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd


def get_landmark(landmark_path):
    df = pd.read_csv(str(landmark_path), header=None)
    coords_list = []
    for i in range(len(df)):
        coords_list.append(df.iloc[i].values)
    return coords_list


def get_crop_info(bbox_path):
    df = pd.read_csv(str(bbox_path), header=None)
    coords_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        coords_list.append(coords)
    return coords_list


speaker_list = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab"]
for speaker in speaker_list:
    data_dir = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
    bbox_dir = Path(f"~/dataset/lip/bbox/{speaker}").expanduser()
    landmark_dir = Path(f"~/dataset/lip/landmark/{speaker}").expanduser()
    save_dir = Path(f"~/dataset/lip/cropped_max_size/{speaker}").expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)

    data_path_list = sorted(list(data_dir.glob("*.mp4")))
    for data_path in tqdm(data_path_list):
        if (save_dir / f"{data_path.stem}.mp4").exists():
            continue
        
        bbox_path = bbox_dir / f"{data_path.stem}.csv"
        if not bbox_path.exists():
            continue
        
        bbox_list = get_crop_info(bbox_path)
        
        edge_list = [np.inf, np.inf, 0, 0]
        for bbox in bbox_list:
            for i in range(4):
                if i <= 1:
                    edge_list[i] = min(edge_list[i], bbox[i])
                else:
                    edge_list[i] = max(edge_list[i], bbox[i])
            
        crop_size = int(max(edge_list[2] - edge_list[0], edge_list[3] - edge_list[1]) // 2 * 2)
        center = [
            (edge_list[2] + edge_list[0]) // 2, 
            (edge_list[3] + edge_list[1]) // 2
        ]
        crop_area = [
            int(center[1] - crop_size / 2),
            int(center[1] + crop_size / 2),
            int(center[0] - crop_size / 2),
            int(center[0] + crop_size / 2),
        ]
        
        movie = cv2.VideoCapture(str(data_path))
        fps = movie.get(cv2.CAP_PROP_FPS)
        n_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_size = crop_size
        out = cv2.VideoWriter(f"{save_dir}/{data_path.stem}.mp4", int(fourcc), fps, (int(output_size), int(output_size)))
        
        while True:
            ret, frame = movie.read()
            if ret == False:
                break
            frame = frame[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]]
            frame = cv2.resize(frame, (output_size, output_size))
            out.write(frame)
        out.release()