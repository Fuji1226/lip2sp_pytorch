import cv2
import dlib
import os

def track_faces(input_video_path, output_directory):
    # 顔検出器の初期化
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")

    # 入力動画のキャプチャ
    cap = cv2.VideoCapture(input_video_path)

    # 保存する動画の設定
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # トラッキング結果を保存するための辞書
    face_trackers = {}

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔検出
        faces = detector(gray)

        for face in faces:
            # 顔の領域を切り取り
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_region = frame[y:y + h, x:x + w]

            # 保存先のディレクトリを作成
            face_id = face_trackers.get(face, len(face_trackers) + 1)
            save_directory = os.path.join(output_directory, f"person_{face_id}")
            os.makedirs(save_directory, exist_ok=True)

            # 顔領域を保存
            frame_path = os.path.join(save_directory, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
            cv2.imwrite(frame_path, face_region)

            # トラッキング用の枠を描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # トラッカーの初期化
            if face not in face_trackers:
                tracker = cv2.TrackerKCF_create()
                face_trackers[face] = tracker
                tracker.init(frame, (x, y, w, h))
            else:
                # トラッキングの更新
                tracker = face_trackers[face]
                success, tracking_result = tracker.update(frame)
                if success:
                    tracking_result = [int(coord) for coord in tracking_result]
                    cv2.rectangle(
                        frame, 
                        (tracking_result[0], tracking_result[1]),
                        (tracking_result[0] + tracking_result[2], tracking_result[1] + tracking_result[3]),
                        (255, 0, 0),
                        2,
                    )

        # 結果を動画として保存
        output_video_path = os.path.join(output_directory, f"output_person_{face_id}.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        out.write(frame)
        out.release()

    cap.release()


if __name__ == "__main__":
    input_video_path = "path/to/your/input_video.mp4"
    output_directory = "path/to/your/output_directory"

    # トラッキングを行い、各顔領域を保存
    track_faces(input_video_path, output_directory)
