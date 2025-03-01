# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## Based on: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/crop_mouth_from_video.py

"""Crop Mouth ROIs from videos for lipreading"""

import argparse
import os
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import transform as tf
from tqdm import tqdm


# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform(
        "similarity", src, dst
    )  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped


def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    cutted_img = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return cutted_img


def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals) + ".png"), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, "w") as fo:
        fo.write("file " + "'" + tmp_dir + "/%0" + str(decimals) + "d.png" + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [
        ffmpeg,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_fn,
        "-q:v",
        "1",
        "-r",
        str(fps),
        "-y",
        "-crf",
        "20",
        target_path,
    ]
    pipe = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return


def load_args(default_config=None):
    parser = argparse.ArgumentParser(
        description="Lipreading Pre-processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument('--video-direc', default=None, help='raw video directory')
    # parser.add_argument('--landmark-direc', default=None, help='landmark directory')
    # parser.add_argument('--filename-path', help='list of detected video and its subject ID')
    # parser.add_argument('--save-direc', default=None, help='the directory of saving mouth ROIs')
    # -- mean face utils
    # parser.add_argument('--mean-face', type=str, help='reference mean face (download from: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)')
    # -- mouthROIs utils
    # parser.add_argument('--speaker', nargs="*", type=str, required=True)
    parser.add_argument(
        "--crop-width", default=96, type=int, help="the width of mouth ROIs"
    )
    parser.add_argument(
        "--crop-height", default=96, type=int, help="the height of mouth ROIs"
    )
    parser.add_argument(
        "--start-idx", default=48, type=int, help="the start of landmark index"
    )
    parser.add_argument(
        "--stop-idx", default=68, type=int, help="the end of landmark index"
    )
    parser.add_argument(
        "--window-margin",
        default=12,
        type=int,
        help="window margin for smoothed_landmarks",
    )
    # parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    # parser.add_argument('--rank', type=int, help='rank id')
    # parser.add_argument('--nshard', type=int, help='number of shards')

    args = parser.parse_args()
    return args


def crop_patch(
    video_pathname,
    landmarks,
    mean_face_landmarks,
    stablePntsIDs,
    STD_SIZE,
    window_margin,
    start_idx,
    stop_idx,
    crop_height,
    crop_width,
):
    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    while True:
        try:
            frame = frame_gen.__next__()  ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        # margin分だけ前のフレームからの平均を計算することで平滑化。
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)  # (68, 2)
            cur_landmarks = q_landmarks.popleft()  # (68, 2)
            cur_frame = q_frame.popleft()  # (H, W, 3)
            # -- affine transformation
            trans_frame, trans = warp_img(
                smoothed_landmarks[stablePntsIDs, :],
                mean_face_landmarks[stablePntsIDs, :],
                cur_frame,
                STD_SIZE,
            )
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(
                cut_patch(
                    trans_frame,
                    trans_landmarks[start_idx:stop_idx],  # 口周辺のランドマーク
                    crop_height // 2,
                    crop_width // 2,
                )
            )

        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(
                    cut_patch(
                        trans_frame,
                        trans_landmarks[start_idx:stop_idx],
                        crop_height // 2,
                        crop_width // 2,
                    )
                )
            return np.array(sequence)
        frame_idx += 1
    return sequence


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(
                landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
            )
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [
            landmarks[valid_frames_idx[0]]
        ] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
            len(landmarks) - valid_frames_idx[-1]
        )
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def get_landmark(landmark_path):
    df = pd.read_csv(str(landmark_path), header=None)
    coords_list = []
    for i in range(len(df)):
        coords_list.append(df.iloc[i].values.reshape(68, 2))
    return coords_list


def main():
    args = load_args()

    # -- mean face utils
    STD_SIZE = (256, 256)
    mean_face_path = Path("~/dataset/lip/20words_mean_face.npy").expanduser()
    mean_face_landmarks = np.load(str(mean_face_path))
    stablePntsIDs = [33, 36, 39, 42, 45]
    ffmpeg_path = "/usr/bin/ffmpeg"

    video_dir = Path("~/dataset/lip/cropped_fps25").expanduser()
    landmark_dir = Path("~/dataset/lip/landmark_fps25").expanduser()
    save_dir = Path("~/dataset/lip/avhubert_preprocess_fps25").expanduser()

    video_dir_list = list(video_dir.glob("*"))
    for video_dir_spk in video_dir_list:
        speaker = video_dir_spk.stem
        if speaker == "F01_kablab_20220930":
            continue
        print(f"speaker = {speaker}")

        video_path_list = list(video_dir_spk.glob("*.mp4"))
        for video_path in tqdm(video_path_list):
            landmark_path = (
                landmark_dir / video_path.parents[0].name / f"{video_path.stem}.csv"
            )

            if (not video_path.exists()) or (not landmark_path.exists()):
                continue

            save_path = save_dir / video_path.parents[0].name / f"{video_path.stem}.mp4"
            if save_path.exists():
                continue
            save_path.parents[0].mkdir(parents=True, exist_ok=True)

            landmarks = get_landmark(str(landmark_path))
            preprocessed_landmarks = landmarks_interpolate(landmarks)

            if not preprocessed_landmarks:
                frame_gen = read_video(str(video_path))
                frames = [
                    cv2.resize(x, (args.crop_width, args.crop_height))
                    for x in frame_gen
                ]
                write_video_ffmpeg(frames, save_path, str(ffmpeg_path))

            sequence = crop_patch(
                str(video_path),
                preprocessed_landmarks,
                mean_face_landmarks,
                stablePntsIDs,
                STD_SIZE,
                window_margin=args.window_margin,
                start_idx=args.start_idx,
                stop_idx=args.stop_idx,
                crop_height=args.crop_height,
                crop_width=args.crop_width,
            )
            write_video_ffmpeg(sequence, save_path, str(ffmpeg_path))


if __name__ == "__main__":
    main()
