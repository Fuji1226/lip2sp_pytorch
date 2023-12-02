from pathlib import Path
import csv
import numpy as np
import face_alignment
import av
from tqdm import tqdm
import torch


def main():
    debug = False
    data_dir = Path('~/tcd_timit_fps25').expanduser()
    data_path_list = list(data_dir.glob('**/*.mp4'))
    if debug:
        data_path_list = data_path_list[:5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)

    for data_path in tqdm(data_path_list):
        save_path = Path(str(data_path).replace('tcd_timit_fps25', 'tcd_timit_fps25_landmark').replace('.mp4', '.csv'))
        if save_path.exists():
            continue
        save_path.parents[0].mkdir(parents=True, exist_ok=True)

        landmark_list = []
        bbox_list = []
        try:
            container = av.open(str(data_path))
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

            with open(str(save_path), "w") as f:
                writer = csv.writer(f)
                for landmark in landmark_list:
                    writer.writerow(landmark)
        except:
            continue


if __name__ == '__main__':
    main()