from pathlib import Path
import os
import csv
import numpy as np
import av
import torch
import face_alignment
from tqdm import tqdm
import cv2
import librosa
from scipy.io.wavfile import write
import subprocess
import argparse

DEBUG = False
SR = 16000
NUM_FRAME_LIMIT_MIN = 50

LANDMARK_SAVE_DIR_NAME = "landmark_split_wide"
BBOX_SAVE_DIR_NAME = "bbox_split_wide"
VIDEO_SAVE_DIR_NAME = "Dataset_split_wide"

if DEBUG:
    LANDMARK_SAVE_DIR_NAME = LANDMARK_SAVE_DIR_NAME + "_debug"
    BBOX_SAVE_DIR_NAME = BBOX_SAVE_DIR_NAME + "_debug"
    VIDEO_SAVE_DIR_NAME = VIDEO_SAVE_DIR_NAME + "_debug"


def crop_frame(frame, speaker):
	if speaker == "chem" or speaker == "hs":
		return frame
	elif speaker == "chess":
		return frame[270:460, 770:1130]
	elif speaker == "dl" or speaker == "eh": 
		return  frame[int(frame.shape[0]*2/4):, int(frame.shape[1]*2/4): ]
		# return  frame[int(frame.shape[0]*3/4):, int(frame.shape[1]*3/4): ]


def save_csv(save_path, data_list):
    with open(str(save_path), "w") as f:
        writer = csv.writer(f)
        for landmark in data_list:
            writer.writerow(landmark)


def save_video(frame_list, wav, data_path, fps, width, height, i, split_count):
    # print("\nsave_video start")
    # print(f"frame_list = {len(frame_list)}")
    # print(f"wav = {wav.shape}")
    # print(f"fps = {fps}, width = {width}, height = {height}")
    # print(f"i = {i}, split_count = {split_count}")
    start_second = (i + 1 - len(frame_list)) / fps
    end_second = (i + 1) / fps
    # print(f"start_second = {start_second}, end_second = {end_second}")
    wav_split = wav[int(SR * start_second):int(SR * end_second)]
    wav_split /= np.max(np.abs(wav_split))
    # print(f"wav_split = {wav_split.shape}")
    audio_save_path = Path(str(data_path).replace("Dataset", VIDEO_SAVE_DIR_NAME).replace(data_path.stem, f"{data_path.stem}_split{split_count}").replace(".mp4", ".wav"))
    audio_save_path.parents[0].mkdir(parents=True, exist_ok=True)
    # print(f"audio_save_path = {audio_save_path}")
    write(str(audio_save_path), rate=16000, data=wav_split)
    # print("wav process done")
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    video_save_path = Path(str(data_path).replace("Dataset", VIDEO_SAVE_DIR_NAME).replace(data_path.stem, f"{data_path.stem}_split{split_count}"))
    video_save_path.parents[0].mkdir(parents=True, exist_ok=True)
    # print(f"video_save_path = {video_save_path}")
    out = cv2.VideoWriter(str(video_save_path), fourcc, fps, (int(width), int(height))) 
    for frame in frame_list:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(frame.shape)
        out.write(frame)
    out.release()
    # print("video process done")
    
    merged_save_path = Path(str(video_save_path).replace(video_save_path.stem, f"{video_save_path.stem}_merged"))
    # print(f"merged_save_path = {merged_save_path}")
    
    cmd = f"ffmpeg -y -i {str(video_save_path)} -r {fps} -i {str(audio_save_path)} -ar {SR} -ac 1 {str(merged_save_path)} -loglevel fatal"
    subprocess.run(cmd, shell=True)
    # print("merge done")
    
    cmd = f"rm {str(video_save_path)} {str(audio_save_path)}"
    subprocess.run(cmd, shell=True)
    # print("delete done")


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-s", "--speaker", nargs="+")
    args = argParser.parse_args()
    
    data_dir = Path("~/Lip2Wav/Dataset_fps25").expanduser()
    speaker_list = args.speaker
    # speaker_list = ["chem", "chess", "dl", "eh", "hs"]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)
    
    splitted_data_dir = Path(str(data_dir).replace("Dataset", VIDEO_SAVE_DIR_NAME))

    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        spk_dir = data_dir / speaker / "intervals"
        data_path_list = list(spk_dir.glob("*/*.mp4"))
        
        splitted_data_dir_spk = splitted_data_dir / speaker
        splitted_data_path_list = list(splitted_data_dir_spk.glob("*/*/*.mp4"))
        splitted_data_path_list = [f"{data_path.parents[0].name}/{str(data_path.stem).split('_')[0]}" for data_path in splitted_data_path_list]
        
        if DEBUG:
            data_path_list = data_path_list[:5]
        
        for data_path in tqdm(data_path_list):
            if f"{data_path.parents[0].name}/{data_path.stem}" in splitted_data_path_list:
                continue
            
            save_path_landmark = Path(str(data_path).replace("Dataset", LANDMARK_SAVE_DIR_NAME).replace(".mp4", ".csv"))
            save_path_bbox = Path(str(data_path).replace("Dataset", BBOX_SAVE_DIR_NAME).replace(".mp4", ".csv"))
            
            save_path_landmark.parents[0].mkdir(parents=True, exist_ok=True)
            save_path_bbox.parents[0].mkdir(parents=True, exist_ok=True)
            
            frame_list = []
            landmark_list = []
            bbox_list = []
            
            # container = av.open(str(data_path))
            # stream = container.streams.video[0]
            # fps = int(stream.average_rate)
            
            # wav, fs = librosa.load(str(data_path), sr=SR)
            
            # split_count = 0

            # for i, frame in enumerate(container.decode(video=0)):
            #     img = frame.to_image()
            #     arr = np.asarray(img)   # (H, W, C)
            #     arr = crop_frame(arr, speaker)
            #     width = arr.shape[1]
            #     height = arr.shape[0]
                
            #     landmarks, landmark_scores, bboxes = fa.get_landmarks(arr, return_bboxes=True, return_landmark_score=True)
                
            #     if landmarks is not None:
            #         max_mean = 0
            #         max_score_idx = 0
            #         for j, score in enumerate(landmark_scores):
            #             score_mean = np.mean(score)
            #             if score_mean > max_mean:
            #                 max_mean = score_mean
            #                 max_score_idx = j

            #         landmark = landmarks[max_score_idx]
            #         bbox = bboxes[max_score_idx][:-1]

            #         coords_list = []
            #         for coords in landmark:
            #             coords_list.append(coords[0])
            #             coords_list.append(coords[1])

            #         frame_list.append(arr)
            #         landmark_list.append(coords_list)
            #         bbox_list.append(bbox)
                
            #     else:
            #         if len(frame_list) < NUM_FRAME_LIMIT_MIN:
            #             continue
                    
            #         save_path_landmark_split = save_path_landmark.parents[0] / f"{data_path.stem}_split{split_count}.csv"
            #         save_path_bbox_split = save_path_bbox.parents[0] / f"{data_path.stem}_split{split_count}.csv"
                    
            #         # print("save_split")
            #         # print(len(frame_list), len(landmark_list), len(bbox_list))
            #         # print(f"save_path_landmark_split = {save_path_landmark_split}")
            #         # print(f"save_path_bbox_split = {save_path_bbox_split}")
                    
            #         save_csv(save_path_landmark_split, landmark_list)
            #         save_csv(save_path_bbox_split, bbox_list)
            #         save_video(frame_list, wav, data_path, fps, width, height, i, split_count)
                    
            #         frame_list = []        
            #         landmark_list = []
            #         bbox_list = []
            #         split_count += 1

            #     print(len(frame_list), split_count)
            
            # if len(frame_list) < NUM_FRAME_LIMIT_MIN:
            #     continue
            
            # save_path_landmark_split = save_path_landmark.parents[0] / f"{data_path.stem}_split{split_count}.csv"
            # save_path_bbox_split = save_path_bbox.parents[0] / f"{data_path.stem}_split{split_count}.csv"
            
            # # print("save last")
            # # print(len(frame_list), len(landmark_list), len(bbox_list))
            # # print(f"save_path_landmark_split = {save_path_landmark_split}")
            # # print(f"save_path_bbox_split = {save_path_bbox_split}")
            
            # save_csv(save_path_landmark_split, landmark_list)
            # save_csv(save_path_bbox_split, bbox_list)
            # save_video(frame_list, wav, data_path, fps, width, height, i, split_count)
            
            try:
                container = av.open(str(data_path))
                stream = container.streams.video[0]
                fps = int(stream.average_rate)
                
                wav, fs = librosa.load(str(data_path), sr=SR)
                
                split_count = 0
                
                for i, frame in enumerate(container.decode(video=0)):
                    img = frame.to_image()
                    arr = np.asarray(img)   # (H, W, C)
                    arr = crop_frame(arr, speaker)
                    width = arr.shape[1]
                    height = arr.shape[0]
                    
                    landmarks, landmark_scores, bboxes = fa.get_landmarks(arr, return_bboxes=True, return_landmark_score=True)
                    
                    if landmarks is not None:
                        max_mean = 0
                        max_score_idx = 0
                        for j, score in enumerate(landmark_scores):
                            score_mean = np.mean(score)
                            if score_mean > max_mean:
                                max_mean = score_mean
                                max_score_idx = j

                        landmark = landmarks[max_score_idx]
                        bbox = bboxes[max_score_idx][:-1]

                        coords_list = []
                        for coords in landmark:
                            coords_list.append(coords[0])
                            coords_list.append(coords[1])

                        frame_list.append(arr)
                        landmark_list.append(coords_list)
                        bbox_list.append(bbox)
                    
                    else:
                        if len(frame_list) < NUM_FRAME_LIMIT_MIN:
                            continue
                        
                        save_path_landmark_split = save_path_landmark.parents[0] / f"{data_path.stem}_split{split_count}.csv"
                        save_path_bbox_split = save_path_bbox.parents[0] / f"{data_path.stem}_split{split_count}.csv"
                        
                        # print("save_split")
                        # print(len(frame_list), len(landmark_list), len(bbox_list))
                        # print(f"save_path_landmark_split = {save_path_landmark_split}")
                        # print(f"save_path_bbox_split = {save_path_bbox_split}")
                        
                        save_csv(save_path_landmark_split, landmark_list)
                        save_csv(save_path_bbox_split, bbox_list)
                        save_video(frame_list, wav, data_path, fps, width, height, i, split_count)
                        
                        frame_list = []        
                        landmark_list = []
                        bbox_list = []
                        split_count += 1

                if len(frame_list) < NUM_FRAME_LIMIT_MIN:
                    continue
                
                save_path_landmark_split = save_path_landmark.parents[0] / f"{data_path.stem}_split{split_count}.csv"
                save_path_bbox_split = save_path_bbox.parents[0] / f"{data_path.stem}_split{split_count}.csv"
                
                # print("save last")
                # print(len(frame_list), len(landmark_list), len(bbox_list))
                # print(f"save_path_landmark_split = {save_path_landmark_split}")
                # print(f"save_path_bbox_split = {save_path_bbox_split}")
                
                save_csv(save_path_landmark_split, landmark_list)
                save_csv(save_path_bbox_split, bbox_list)
                save_video(frame_list, wav, data_path, fps, width, height, i, split_count)
                    
            except:
                print(f"error: {data_path}")
                continue
                
        
if __name__ == "__main__":
    main()