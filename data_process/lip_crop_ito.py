import os
from pathlib import Path
import glob
import cv2
import dlib
import numpy as np
from tqdm import tqdm

speaker = "F01_kablab_20220930"
data_root = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
save_dir = f"/home/usr4/r70264c/dataset/lip/lip_cropped/{speaker}"
txt_path = f"/home/usr4/r70264c/dataset/lip/cropped_error_data_{speaker}.txt"

debug = False

# 口唇のランドマーク検出
def Lip_Cropping(frame, det):
    predicter_Path = "/home/usr4/r70264c/lip2sp_pytorch/shape_predictor_68_face_landmarks.dat"     # 変えてください
    predictor = dlib.shape_predictor(predicter_Path)
    shape = predictor(frame, det)
    shapes = []
    for shape_point_count in range(shape.num_parts):
        shape_point = shape.part(shape_point_count)
        if shape_point_count >= 48: # LIP
            shapes += [[shape_point.x, shape_point.y]]

    # 平均を中心とする
    mouth_center = np.mean(shapes, axis=0).astype('int')

    # 口唇部分のランドマークの左端と右端から切り取り範囲を決定
    left_point = shapes[0]
    right_point = shapes[6]
    im_size = right_point[0] - left_point[0]

    # はみ出ないように10%分余裕を持たせる
    im_size += int(im_size * 0.1)

    return im_size, mouth_center


def main():
    os.makedirs(save_dir, exist_ok=True)

    print(f"speaker = {speaker}")
    datasets_path = sorted(list(data_root.glob("*.mp4")))

    for data_path in tqdm(datasets_path, total=len(datasets_path)):
        data_name = Path(data_path)
        data_name = data_name.stem      # 保存する口唇動画の名前に使用
    
        # 既にある場合はスルー
        check_path = Path(f"{save_dir}/{data_name}_crop.mp4")
        if check_path.exists():
            continue

        try:
            # 動画読み込み
            movie = cv2.VideoCapture(str(data_path))
            fps    = movie.get(cv2.CAP_PROP_FPS)
            height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width  = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
            if movie.isOpened() == True:
                ret, frame = movie.read()
            else:
                ret = False

            # 顔検出
            detector = dlib.get_frontal_face_detector()
            dets = detector(frame, 1)

            det = dets[0]
            
            # 口唇抽出
            im_size, mouth_center = Lip_Cropping(frame, det)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            os.makedirs(save_dir, exist_ok=True)
            out = cv2.VideoWriter(f"{save_dir}/{data_name}_crop.mp4", int(fourcc), fps, (int(im_size), int(im_size)))
            if movie.isOpened() == True:
                ret, frame = movie.read()
            else:
                ret = False

            mouth_area = [
                mouth_center[1]-im_size//2,
                mouth_center[1]+im_size//2,
                mouth_center[0]-im_size//2,
                mouth_center[0]+im_size//2
            ]
            
            # 各フレームに対して口唇の中心位置を計算し直し、1フレーム目で計算したim_sizeの大きさで切り取っていく
            while ret:
                out.write(frame[mouth_area[0]:mouth_area[1], mouth_area[2]:mouth_area[3]])
                ret,frame = movie.read()
                try:
                    dets = detector(frame, 1)
                    det = dets[0]
                    _, mouth_center = Lip_Cropping(frame, det)
                    mouth_area = [
                        mouth_center[1]-im_size//2,
                        mouth_center[1]+im_size//2,
                        mouth_center[0]-im_size//2,
                        mouth_center[0]+im_size//2
                    ]
                except:
                    mouth_area = mouth_area
            out.release()
        
        except:
            # できないやつをスキップし，txtデータに書き込んでおく
            with open(txt_path, mode="a") as f:
                f.write(f"{data_name}\n")
        
        if debug:
            break


if __name__ == "__main__":
    main()