from pathlib import Path
import os
import csv
import torch
import torchvision
import numpy as np
import face_alignment
import av
from tqdm import tqdm


data_dir = Path("~/lrw/lipread_mp4").expanduser()
save_dir = Path("~/lrw").expanduser()
debug = False
debug_iter = 0

if debug:
    save_dir_landmark = save_dir / "landmark_debug"
    save_dir_bbox = save_dir / "bbox_debug"
else:
    save_dir_landmark = save_dir / "landmark"
    save_dir_bbox = save_dir / "bbox"


def main():
    file_list = []
    for curdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mp4"):
                file_list.append(os.path.join(curdir, file))
            
            # if debug:
            #     if len(file_list) > 10:
            #         break

    file_list = sorted(list(file_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)

    iter_cnt = 0
    for path in tqdm(file_list):
        path = Path(path)
        save_path_landmark = save_dir_landmark / path.parents[1].name / path.parents[0].name
        save_path_bbox = save_dir_bbox / path.parents[1].name / path.parents[0].name
        os.makedirs(save_path_landmark, exist_ok=True)
        os.makedirs(save_path_bbox, exist_ok=True)

        if Path(str(f"{save_path_landmark}/{path.stem}.csv")).exists() and Path(str(f"{save_path_bbox}/{path.stem}.csv")).exists():
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

            with open(str(f"{save_path_landmark / path.stem}.csv"), "w") as f:
                writer = csv.writer(f)
                for landmark in landmark_list:
                    writer.writerow(landmark)

            with open(str(f"{save_path_bbox / path.stem}.csv"), "w") as f:
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