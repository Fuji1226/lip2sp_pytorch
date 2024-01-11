import torchlm
from torchlm.runtime import faceboxesv2_ort, pipnet_ort
from pathlib import Path
import cv2
from tqdm import tqdm


def write_image(
        image,
        landmarks,
        bboxes,
):
    landmark = landmarks[0] # (68, 2)
    bbox = bboxes[0]
    for i in range(landmark.shape[0]):
        x, y = landmark[i].astype(int).to_list()
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, f"{i}", )

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Hello, OpenCV!"
    position = (50, 50)  # テキストの位置 (x, y)
    font_scale = 1
    font_color = (0, 0, 255)  # テキストの色 (BGR形式)
    font_thickness = 2
    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)




def main():
    torchlm.runtime.bind(faceboxesv2_ort())
    torchlm.runtime.bind(
        pipnet_ort(
            onnx_path="/Users/minami/lip2sp_pytorch/dev/landmark_detector_resnet101.onnx",
            num_nb=10,
            num_lms=68,
            input_size=256,
            net_stride=32,
            meanface_type="300w",
        )
    )

    data_dir = Path("/Users/minami/dataset/lip/cropped_fps25")
    data_path_list = list(data_dir.glob("**/*.mp4"))

    for data_path in tqdm(data_path_list):
        cap = cv2.VideoCapture(str(data_path))
        new_video = []
        while True:
            ret, image = cap.read()
            if not ret:
                break
            landmarks, bboxes = torchlm.runtime.forward(image)
            image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
            image = torchlm.utils.draw_landmarks(image, landmarks=landmarks, text=True, font=0.5)
            new_video.append(image)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックを選択
        frame_rate = 25  # フレームレート（必要に応じて調整）
        frame_width = new_video[0].shape[0]  # フレームの幅（必要に応じて調整）
        frame_height = new_video[0].shape[1]  # フレームの高さ（必要に応じて調整）
        save_path = Path(str(data_path).replace("cropped_fps25", "cropped_fps25_torchlm_test"))
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(save_path), fourcc, frame_rate, (frame_width, frame_height))

        for image in new_video:
            out.write(image)
        out.release()


if __name__ == "__main__":
    main()