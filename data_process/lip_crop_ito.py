"""
動画から口唇動画を作成するところをやってみました
lip2sp/notebook/lip.ipynbを参考にしています

下の実行手順でやっていただければ、lip_croppedに口唇動画が入ると思います

実行手順
1. パスの変更
    data_root, save_dir, txt_pathを変更してください
        data_root : 顔のデータがあるところ(元データ)
        save_dir : 口唇部分を切り取った動画の保存先
        txt_path : 口唇の切り取りをミスる時があるので,ミスったデータを記録しておくファイル(事前に作っておいてください)
    
2. 切り取るサイズの変更
    Lip_croppingのim_sizeを変更してください
    ここで切り取る範囲を設定しています

3. predicter_Pathの変更
    Lip_croppingのpredicter_Pathを変更してください
    shape_predicterがない場合は
    http://dlib.net/files/
    からshape_predictor_68_face_landmarks.dat.bz2を事前にダウンロードし,そこまでのパスを設定してください
    これで顔認識をする感じです

4. 実行
"""

import os
from pathlib import Path
import glob
import cv2
import dlib
import numpy as np
from tqdm import tqdm

speaker = "M03_kablab"
data_root = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
save_dir = f"/home/usr4/r70264c/dataset/lip/lip_cropped/{speaker}"
txt_path = f"/home/usr4/r70264c/dataset/lip/cropped_error_data_{speaker}.txt"
corpus = "ATR"
start_num = 0

debug = False
debug_iter = 5

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
    im_size = right_point[0] - left_point[0] + 10

    return im_size, mouth_center


def main():
    os.makedirs(save_dir, exist_ok=True)
    print(f"speaker = {speaker}, corpus = {corpus}, start_num = {start_num}")
    if corpus == "ATR":
        datasets_path = sorted(glob.glob(f"{data_root}/ATR*.mp4")) 
    elif corpus == "BASIC5000":
        datasets_path = sorted(glob.glob(f"{data_root}/BASIC5000*.mp4")) 
    elif corpus == "balanced":
        datasets_path = sorted(glob.glob(f"{data_root}/balanced*.mp4")) 

    if start_num > 0:
        datasets_path = datasets_path[start_num:]

    iter_cnt = 0
    for data_path in tqdm(datasets_path, total=len(datasets_path)):
        iter_cnt += 1
        data_name = Path(data_path)
        data_name = data_name.stem      # 保存する口唇動画の名前に使用
    
        # 既にある場合はスルー
        check_path = Path(f"{save_dir}/{data_name}_crop.mp4")
        if check_path.exists():
            continue
        
        try:
            # 動画読み込み
            movie = cv2.VideoCapture(data_path)
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
            
            # 各フレームに対して口唇の中心位置を計算し直し、1フレーム目で計算した範囲で切り取る
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
            continue
            
        if debug:
            if iter_cnt > debug_iter:
                break


if __name__ == "__main__":
    main()