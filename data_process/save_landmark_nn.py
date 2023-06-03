from pathlib import Path
import os
import csv
import torch
import torchvision
import numpy as np
import face_alignment
import av
import time
from tqdm import tqdm
import numpy as np


debug = False
debug_iter = 5

speaker = "M01_kablab"
data_root = Path(f"~/dataset/lip/cropped_max_size/{speaker}").expanduser()

if data_root.parents[0].name == "cropped":
    dir_name_landmark = "landmark"
    dir_name_bbox = "bbox"
elif data_root.parents[0].name == "face_aligned":
    dir_name_landmark = "landmark_aligned"
    dir_name_bbox = "bbox_aligned"
elif data_root.parents[0].name == "cropped_fps25":
    dir_name_landmark = "landmark_fps25"
    dir_name_bbox = "bbox_fps25"
elif data_root.parents[0].name == "cropped_max_size":
    dir_name_landmark = "landmark_cropped_max_size"
    dir_name_bbox = "bbox_cropped_max_size"

if debug:
    save_dir_landmark = Path(f"~/dataset/lip/{dir_name_landmark}_debug/{speaker}").expanduser()
    save_dir_bbox = Path(f"~/dataset/lip/{dir_name_bbox}_debug/{speaker}").expanduser()
else:
    save_dir_landmark = Path(f"~/dataset/lip/{dir_name_landmark}/{speaker}").expanduser()
    save_dir_bbox = Path(f"~/dataset/lip/{dir_name_bbox}/{speaker}").expanduser()

os.makedirs(save_dir_landmark, exist_ok=True)
os.makedirs(save_dir_bbox, exist_ok=True)


def main():
    print(f"speaker = {speaker}")
    data_path = sorted(list(data_root.glob("*.mp4")))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)

    iter_cnt = 0
    for path in tqdm(data_path):
        if Path(str(f"{save_dir_landmark}/{path.stem}.csv")).exists() and Path(str(f"{save_dir_bbox}/{path.stem}.csv")).exists():
            continue

        landmark_list = []
        bbox_list = []

        try:
            container = av.open(str(path))
            for frame in container.decode(video=0):
                img = frame.to_image()
                arr = np.asarray(img)
                landmarks, landmark_scores, bboxes = fa.get_landmarks(arr, return_bboxes=True, return_landmark_score=True)

                max_mean = 0
                max_score_idx = 0
                for i, score in enumerate(landmark_scores):
                    score_mean = np.mean(score)
                    if score_mean > max_mean:
                        max_mean = score_mean
                        max_score_idx = i

                landmark = landmarks[max_score_idx]
                bbox = bboxes[max_score_idx][:-1]

                coords_list = []
                for coords in landmark:
                    coords_list.append(coords[0])
                    coords_list.append(coords[1])

                landmark_list.append(coords_list)
                bbox_list.append(bbox)

            total_frames = container.streams.video[0].frames
            assert total_frames == len(landmark_list) 
            assert total_frames == len(bbox_list)

            with open(str(f"{save_dir_landmark}/{path.stem}.csv"), "w") as f:
                writer = csv.writer(f)
                for landmark in landmark_list:
                    writer.writerow(landmark)

            with open(str(f"{save_dir_bbox}/{path.stem}.csv"), "w") as f:
                writer = csv.writer(f)
                for bbox in bbox_list:
                    writer.writerow(bbox)

        except:
            continue

        iter_cnt += 1
        if debug:
            if iter_cnt > debug_iter:
                break


if __name__ == "__main__":
    main()