import cv2
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def calculate_histogram_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def split_and_combine(input_path, output_path_template, segment_duration=10):
    video_clip = VideoFileClip(input_path)
    
    total_duration = video_clip.duration
    start_time = 0

    segment_index = 1
    while start_time < total_duration:
        end_time = min(start_time + segment_duration, total_duration)

        # 動画と音声を分割せずにセグメントごとに書き出し
        segment_clip = video_clip.subclip(start_time, end_time)
        output_path = output_path_template.format(segment_index)
        segment_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        start_time += segment_duration
        segment_index += 1


def main():
    video_path = Path("./videos/底辺グループYouTuberあるある.mp4")
    cap = cv2.VideoCapture(str(video_path))
    save_dir = Path("./shot_boundary")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    ret, prev_frame = cap.read()
    prev_hist = calculate_histogram(prev_frame)
    cnt = 0
    dist_thres = 0.5
    num_frame_thres = fps * 10
    frame_list = []
    frame_list.append(prev_frame)
    
    video_clip = VideoFileClip(str(video_path))
    total_duration = video_clip.duration
    start_time = 0
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
        current_hist = calculate_histogram(current_frame)
        dist = calculate_histogram_distance(prev_hist, current_hist)
        if dist > dist_thres or len(frame_list) > num_frame_thres:
            save_path = save_dir / video_path.stem / f"{cnt}.mp4"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            end_time = min(start_time + (len(frame_list) / fps), total_duration)
            segment_clip = video_clip.subclip(start_time, end_time)
            segment_clip.write_videofile(str(save_path), codec="libx264", audio_codec="aac")
            frame_list = []
            cnt += 1
            start_time = end_time
        frame_list.append(current_frame)
        prev_hist = current_hist
        prev_frame = current_frame
        
    
if __name__ == "__main__":
    main()