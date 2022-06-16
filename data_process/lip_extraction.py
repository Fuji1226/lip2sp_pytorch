"""
動画から口唇動画を作成するところをやってみました
lip2sp/notebook/lip.ipynbを参考にしています

下の実行手順でやっていただければ、lip_croppedに口唇動画が入ると思います

実行手順
/Users/minami/dataset/lip 下にlip_croppedディレクトリを作成
-> /Users/minami/dataset/lip/lip_cropped

31行目のsave_dirをlip_croppedまでのパスに変更

44行目のpredicter_Pathを変更（shape_predictor_68_face_landmarks.datまでのパス）

"""

import os
from pathlib import Path
import glob
import cv2
import dlib
import numpy as np


def get_datasetroot():
    ret = Path("~", "dataset")

    ret = ret.expanduser()

    ret.mkdir(exist_ok=True, parents=True)

    return ret

def get_data_directory():
    """return SAVEDIR"""
    return os.path.join(get_datasetroot(), "lip", "cropped")



data_root = Path(get_data_directory()).expanduser()     # /Users/minami/dataset/lip/cropped
save_dir = "/Users/fujitanaoaki/Desktop/lip2sp_pytorch/fujita/lip_cropped"      # 変えてください



# 口唇のランドマーク検出
def Lip_Cropping(frame, det):
    im_size = (96, 128)
    predicter_Path = "/Users/fujitanaoaki/Desktop/lip2sp_pytorch/fujita/shape_predictor_68_face_landmarks.dat"     # 変えてください
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
    datasets = data_root / "F01_kablab"     # /Users/minami/dataset/lip/cropped/F01_kablab
    # F01_kablab内のmp4までのパスを取得。
    datasets_path = sorted(glob.glob(f"{datasets}/*.mp4")) 

    for data_path in datasets_path:
        # print(data_path)
        data_name = Path(data_path)
        data_name = data_name.stem      # 保存する口唇動画の名前に使用
        # print(data_name)
        
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
        assert len(dets) == 1
        det = dets[0]
        face_area = [det.top(), det.bottom(), det.left(), det.right()]
        face = frame[face_area[0]:face_area[1], face_area[2]:face_area[3]]
        
        # 口唇抽出
        im_size, mouth_area = Lip_Cropping(frame, det)
        breakpoint()
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(f"{data_name}.mp4", int(fourcc), fps, (int(im_size[1]), int(im_size[0])))
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