"""
動画全体から口唇部分のみを切り取り
speakerを変更してください
かなり長いのでito frontで複数cpuを予約し,並列で実行するのがおすすめです
"""
import os
from pathlib import Path
import glob
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import csv


debug = False
num_start = 3500
num_end = 4000

speaker = "F01_kablab"
data_root = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
txt_path = Path(f"~/dataset/lip/landmark_error_{speaker}.txt").expanduser()

if debug:
    save_dir = Path(f"~/dataset/lip/landmark_debug/{speaker}").expanduser()
else:
    save_dir = Path(f"~/dataset/lip/landmark/{speaker}").expanduser()


def detect_landmark(frame, det, landmark_list):
    predicter_Path = "/home/usr4/r70264c/lip2sp_pytorch/shape_predictor_68_face_landmarks.dat"     # 変えてください
    predictor = dlib.shape_predictor(predicter_Path)
    shape = predictor(frame, det)
    
    all_landmark = []
    for shape_point_count in range(shape.num_parts):
        shape_point = shape.part(shape_point_count)
        all_landmark.append(shape_point.x)
        all_landmark.append(shape_point.y)

    landmark_list.append(all_landmark)
    return landmark_list


def main():
    os.makedirs(save_dir, exist_ok=True)

    print(f"speaker = {speaker}, num_start = {num_start}, num_end = {num_end}")
    datasets_path = sorted(list(data_root.glob("*.mp4")))[num_start:num_end]

    for data_path in tqdm(datasets_path, total=len(datasets_path)):
        data_name = Path(data_path)
        data_name = data_name.stem      # 保存する口唇動画の名前に使用
    
        # 既にある場合はスルー
        check_path = save_dir / f"{data_name}_landmark.csv"
        if check_path.exists():
            continue

        try:
            # 動画読み込み
            movie = cv2.VideoCapture(str(data_path))
            n_frame = movie.get(cv2.CAP_PROP_FRAME_COUNT)
            if movie.isOpened() == False:
                break

            iter_cnt = 0
            landmark_list = []

            while True:
                ret, frame = movie.read()
                if ret == False:
                    break

                # 顔検出
                detector = dlib.get_frontal_face_detector()
                dets = detector(frame, 1)
                det = dets[0]

                # ランドマークを保持
                landmark_list = detect_landmark(frame, det, landmark_list)
                    
                iter_cnt += 1
            assert n_frame  == len(landmark_list)
            
            with open(str(f"{save_dir}/{data_name}_landmark.csv"), "w") as f:
                writer = csv.writer(f)
                for landmark in landmark_list:
                    writer.writerow(landmark)
    
        except:
            # できないやつをスキップし，txtデータに書き込んでおく
            with open(str(txt_path), "a") as f:
                f.write(f"{data_name}\n")
        
        if debug:
            break


if __name__ == "__main__":
    main()