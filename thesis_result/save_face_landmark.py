from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


speaker = "F01_kablab"
landmark_dir = Path(f"~/dataset/lip/landmark/{speaker}").expanduser()
landmark_dir_aligned = Path(f"~/dataset/lip/landmark_aligned_for_visualize/{speaker}").expanduser()
data_dir = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
data_dir_aligned = Path(f"~/dataset/lip/face_aligned_for_visualize/{speaker}").expanduser()
save_dir_orig = Path(f"~/lip2sp_pytorch/check/face_landmark/{speaker}/orig").expanduser()
save_dir_aligned = Path(f"~/lip2sp_pytorch/check/face_landmark/{speaker}/aligned").expanduser()


def save_face_landmark_img_face_align(landmark_path, face_path, save_dir):
    os.makedirs(save_dir / "images_face" / face_path.stem, exist_ok=True)

    df = pd.read_csv(str(landmark_path), header=None)
    landmark_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        landmark_list.append([coords[0::2], coords[1::2]])
    landmark = np.array(landmark_list)

    cap = cv2.VideoCapture(str(face_path))
    i = 0

    while True:
        ret, frame = cap.read()
        if ret:
            coords_x = landmark[i, 0, :]
            coords_y = landmark[i, 1, :]
            for j in range(coords_x.shape[-1]):
                cv2.drawMarker(
                    img=frame, 
                    position=(int(coords_x[j]), int(coords_y[j])), 
                    color=(255, 0, 0), 
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=5,
                    thickness=1,
                )
            left_eye_center = [np.mean(coords_x[36:42]).astype("int"), np.mean(coords_y[36:42]).astype("int")]
            right_eye_center = [np.mean(coords_x[42:48]).astype("int"), np.mean(coords_y[42:48]).astype("int")]
            mid = [(left_eye_center[i] + right_eye_center[i]) // 2 for i in range(len(left_eye_center))]
            cv2.drawMarker(
                img=frame, 
                position=(left_eye_center[0], left_eye_center[1]), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.drawMarker(
                img=frame, 
                position=(right_eye_center[0], right_eye_center[1]), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.drawMarker(
                img=frame, 
                position=(mid[0], mid[1]), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.line(
                img=frame,
                pt1=(left_eye_center[0], left_eye_center[1]),
                pt2=(right_eye_center[0], right_eye_center[1]),
                color=(0, 0, 255),
                thickness=1,
            )

            cv2.imwrite(f"{save_dir}/images_face/{face_path.stem}/{i}.png", frame)
            i += 1
        else:
            break


def save_face_landmark_img_lip_crop(landmark_path, face_path, save_dir, margin):
    os.makedirs(save_dir / f"images_lip_{margin}" / face_path.stem, exist_ok=True)

    df = pd.read_csv(str(landmark_path), header=None)
    landmark_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        landmark_list.append([coords[0::2], coords[1::2]])
    landmark = np.array(landmark_list)

    cap = cv2.VideoCapture(str(face_path))
    i = 0

    while True:
        ret, frame = cap.read()
        if ret:
            coords_x = landmark[i, 0, :]
            coords_y = landmark[i, 1, :]
            for j in range(coords_x.shape[-1]):
                cv2.drawMarker(
                    img=frame, 
                    position=(int(coords_x[j]), int(coords_y[j])), 
                    color=(255, 0, 0), 
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=5,
                    thickness=1,
                )
            lip_center = [np.mean(coords_x[48:]).astype("int"), np.mean(coords_y[48:]).astype("int")]
            length = coords_x[54] - coords_x[48]
            crop_size = length + length * margin
            left_pt = [lip_center[0] - crop_size // 2, lip_center[1] - crop_size // 2]
            right_pt = [lip_center[0] + crop_size // 2, lip_center[1] + crop_size // 2]

            cv2.drawMarker(
                img=frame, 
                position=(int(lip_center[0]), int(lip_center[1])), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.rectangle(
                img=frame,
                pt1=(int(left_pt[0]), int(left_pt[1])),
                pt2=(int(right_pt[0]), int(right_pt[1])),
                color=(0, 0, 255),
                thickness=1,
            )

            cv2.imwrite(f"{save_dir}/images_lip_{margin}/{face_path.stem}/{i}.png", frame)
            i += 1
        else:
            break

    
def save_face_landmark_video(landmark_path, face_path, save_dir):
    os.makedirs(save_dir / "video", exist_ok=True)

    df = pd.read_csv(str(landmark_path), header=None)
    landmark_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        landmark_list.append([coords[0::2], coords[1::2]])
    landmark = np.array(landmark_list)

    cap = cv2.VideoCapture(str(face_path))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(face_path)
    print(width, height, fps)
    out = cv2.VideoWriter(f"{save_dir}/video/{face_path.stem}.mp4", int(fourcc), int(fps), (int(width), int(height)))
    i = 0

    while True:
        ret, frame = cap.read()
        if ret:
            coords_x = landmark[i, 0, :]
            coords_y = landmark[i, 1, :]
            for j in range(coords_x.shape[-1]):
                cv2.drawMarker(
                    img=frame, 
                    position=(int(coords_x[j]), int(coords_y[j])), 
                    color=(255, 0, 0), 
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=5,
                    thickness=1,
                )

            out.write(frame)
            i += 1
        else:
            break

    out.release()


def save_face_landmark_video_line(landmark_path, face_path, save_dir):
    os.makedirs(save_dir / "video_line", exist_ok=True)

    df = pd.read_csv(str(landmark_path), header=None)
    landmark_list = []
    for i in range(len(df)):
        coords = df.iloc[i].values
        landmark_list.append([coords[0::2], coords[1::2]])
    landmark = np.array(landmark_list)

    cap = cv2.VideoCapture(str(face_path))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(f"{save_dir}/video_line/{face_path.stem}.mp4", int(fourcc), int(fps), (int(width), int(height)))
    i = 0

    while True:
        ret, frame = cap.read()
        if ret:
            coords_x = landmark[i, 0, :]
            coords_y = landmark[i, 1, :]
            for j in range(coords_x.shape[-1]):
                cv2.drawMarker(
                    img=frame, 
                    position=(int(coords_x[j]), int(coords_y[j])), 
                    color=(255, 0, 0), 
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=5,
                    thickness=1,
                )
            left_eye_center = [np.mean(coords_x[36:42]).astype("int"), np.mean(coords_y[36:42]).astype("int")]
            right_eye_center = [np.mean(coords_x[42:48]).astype("int"), np.mean(coords_y[42:48]).astype("int")]
            mid = [(left_eye_center[i] + right_eye_center[i]) // 2 for i in range(len(left_eye_center))]
            cv2.drawMarker(
                img=frame, 
                position=(left_eye_center[0], left_eye_center[1]), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.drawMarker(
                img=frame, 
                position=(right_eye_center[0], right_eye_center[1]), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.drawMarker(
                img=frame, 
                position=(mid[0], mid[1]), 
                color=(0, 0, 255), 
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            cv2.line(
                img=frame,
                pt1=(left_eye_center[0], left_eye_center[1]),
                pt2=(right_eye_center[0], right_eye_center[1]),
                color=(0, 0, 255),
                thickness=1,
            )

            out.write(frame)
            i += 1
        else:
            break

    out.release()


def main():
    landmark_path = sorted(list(landmark_dir.glob("*.csv")))
    exp_path_landmark = landmark_path[0]
    exp_path_face = data_dir / f"{exp_path_landmark.stem}.mp4"
    save_face_landmark_img_face_align(exp_path_landmark, exp_path_face, save_dir_orig)
    save_face_landmark_img_lip_crop(exp_path_landmark, exp_path_face, save_dir_orig, margin=0.3)
    save_face_landmark_img_lip_crop(exp_path_landmark, exp_path_face, save_dir_orig, margin=0.8)
    save_face_landmark_video(exp_path_landmark, exp_path_face, save_dir_orig)
    save_face_landmark_video_line(exp_path_landmark, exp_path_face, save_dir_orig)
    
    landmark_path_aligned = sorted(list(landmark_dir_aligned.glob("*.csv")))
    exp_path_landmark_alinged = landmark_path_aligned[0]
    exp_path_face_aligned = data_dir_aligned / f"{exp_path_landmark_alinged.stem}.mp4"
    save_face_landmark_img_face_align(exp_path_landmark_alinged, exp_path_face_aligned, save_dir_aligned)
    save_face_landmark_img_lip_crop(exp_path_landmark_alinged, exp_path_face_aligned, save_dir_aligned, margin=0.3)
    save_face_landmark_img_lip_crop(exp_path_landmark_alinged, exp_path_face_aligned, save_dir_aligned, margin=0.8)
    save_face_landmark_video(exp_path_landmark_alinged, exp_path_face_aligned, save_dir_aligned)
    save_face_landmark_video_line(exp_path_landmark_alinged, exp_path_face_aligned, save_dir_aligned)


if __name__ == "__main__":
    main()