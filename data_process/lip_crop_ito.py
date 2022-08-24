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
from sqlalchemy import desc


data_root = Path("~/dataset/lip/cropped/F01_kablab").expanduser()
save_dir = "/home/usr4/r70264c/dataset/lip/lip_cropped_9696/F01_kablab"      # 変えてください
txt_path = "/home/usr4/r70264c/dataset/lip/cropped_error_data3.txt"      # エラーしたデータの書き込み用(事前に作成してください)


# 口唇のランドマーク検出
def Lip_Cropping(frame, det):
    im_size = (96, 96)  # 変更
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

    # mouth_centerを中心に、im_sizeの大きさにする
    mouth_area = [mouth_center[1]-im_size[0]//2,
                  mouth_center[1]+im_size[0]//2,
                  mouth_center[0]-im_size[1]//2,
                  mouth_center[0]+im_size[1]//2]
    
    # 切り取り
    # mouth = frame[mouth_area[0]:mouth_area[1], mouth_area[2]:mouth_area[3]]
    return im_size, mouth_area


def main():
    os.makedirs(save_dir, exist_ok=True)
    datasets = data_root
    # F01_kablab内のmp4までのパスを取得。
    datasets_path = sorted(glob.glob(f"{datasets}/*.mp4")) 

    for data_path in datasets_path:
        # print(data_path)
        data_name = Path(data_path)
        data_name = data_name.stem      # 保存する口唇動画の名前に使用
    
        # 既にある場合はスルー
        check_path = Path(f"{save_dir}/{data_name}_crop.mp4")
        if check_path.exists():
            continue
        
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
        # assert len(dets) == 1

        ########################################
        # できないやつをスキップし，txtデータに書き込んでおく
        if len(dets) != 1:
            with open(txt_path, mode="a") as f:
                f.write(f"{data_name}\n")
            continue
        ########################################

        det = dets[0]
        face_area = [det.top(), det.bottom(), det.left(), det.right()]
        face = frame[face_area[0]:face_area[1], face_area[2]:face_area[3]]
        
        # 口唇抽出
        im_size, mouth_area = Lip_Cropping(frame, det)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(f"{save_dir}/{data_name}_crop.mp4", int(fourcc), fps, (int(im_size[1]), int(im_size[0])))
        if movie.isOpened() == True:
            ret, frame = movie.read()
        else:
            ret = False
        while ret:
            out.write(frame[mouth_area[0]:mouth_area[1], mouth_area[2]:mouth_area[3]])
            ret,frame = movie.read()
        out.release()

    



if __name__ == "__main__":
    main()