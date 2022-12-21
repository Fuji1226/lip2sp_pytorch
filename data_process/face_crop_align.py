from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

speaker = "F01_kablab"
landmark_dir = Path(f"~/dataset/lip/landmark/{speaker}").expanduser()
data_dir = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
desired_left_eye = (0.35, 0.35)
desired_face_size = 256
debug = True
debug_iter = 5

if debug:
    dir_name = "face_aligned_debug"
else:
    dir_name = "face_aligned"
save_dir = Path(f"~/dataset/lip/{dir_name}/{speaker}").expanduser()
os.makedirs(save_dir, exist_ok=True)


class FaceAligner:
    def __init__(self, desired_left_eye, desired_face_width, desired_face_height):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

    def align(self, frame, landmark):
        left_eye_coords = landmark[int(36 * 2) : int(42 * 2)]
        right_eye_coords = landmark[int(42 * 2) : int(48 * 2)]
        left_eye_center = [np.mean(left_eye_coords[0::2]).astype("int"), np.mean(left_eye_coords[1::2]).astype("int")]
        right_eye_center = [np.mean(right_eye_coords[0::2]).astype("int"), np.mean(right_eye_coords[1::2]).astype("int")]

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desiredDist = (desired_right_eye_x - self.desired_left_eye[0])
        desiredDist *= self.desired_face_width
        scale = desiredDist / dist

        eyes_center = (
            int((left_eye_center[0] + right_eye_center[0]) // 2),
            int((left_eye_center[1] + right_eye_center[1]) // 2),
        )

        matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tx = self.desired_face_width * 0.5
        ty = self.desired_face_height * self.desired_left_eye[1]
        matrix[0, 2] += (tx - eyes_center[0])
        matrix[1, 2] += (ty - eyes_center[1])

        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_CUBIC)
        return output


def get_landmark(landmark_path):
    df = pd.read_csv(str(landmark_path), header=None)
    coords_list = []
    for i in range(len(df)):
        coords_list.append(df.iloc[i].values)
    return coords_list


def crop(aligner, data_path, landmark_path, save_dir):
    coords_list = get_landmark(landmark_path)
    movie = cv2.VideoCapture(str(data_path))
    width = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = movie.get(cv2.CAP_PROP_FPS)
    n_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
    assert n_frame == len(coords_list)

    print(width, height)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(f"{save_dir}/{data_path.stem}.mp4", int(fourcc), fps, (int(desired_face_size), int(desired_face_size)))

    iter_cnt = 0
    while True:
        ret, frame = movie.read()
        if ret == False:
            break

        frame_aligned = aligner.align(frame, coords_list[iter_cnt])
        iter_cnt += 1

        out.write(frame_aligned)

    out.release()


def main():
    print(f"speaker = {speaker}")
    aligner = FaceAligner(desired_left_eye, desired_face_size, desired_face_size)

    iter_cnt = 0
    landmark_path = sorted(list(landmark_dir.glob("*.csv")))
    for landmark_path in tqdm(landmark_path):
        data_path = Path(f"{data_dir / landmark_path.stem}.mp4")
        crop(aligner, data_path, landmark_path, save_dir)

        iter_cnt += 1
        if debug:
            if iter_cnt > debug_iter:
                break


if __name__ == "__main__":
    main()