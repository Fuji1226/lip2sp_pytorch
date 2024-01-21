import cv2
import numpy as np
from pathlib import Path


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def calculate_histogram_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def detect_shot_boundaries(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    cnt = 0

    if not ret:
        print("Error reading video file.")
        return

    prev_hist = calculate_histogram(prev_frame)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_hist = calculate_histogram(frame)
        distance = calculate_histogram_distance(prev_hist, current_hist)

        if distance > threshold:
            print("Shot boundary detected!")
            save_path = Path("./shot_boundary")
            save_path = save_path / str(cnt)
            save_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path / "prev_frame.png"), prev_frame)
            cv2.imwrite(str(save_path / "frame.png"), frame)
            cnt += 1

        prev_hist = current_hist
        prev_frame = frame

    cap.release()
    

if __name__ == "__main__":
    video_path = "W⧸X⧸Y - Tani Yuuki (Official Lyric Video).mp4"
    detect_shot_boundaries(video_path)
