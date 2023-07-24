from pathlib import Path
import os
import csv
import numpy as np
import av
import torch
import face_alignment
from tqdm import tqdm


def crop_frame(frame, speaker):
	if speaker == "chem" or speaker == "hs":
		return frame
	elif speaker == "chess":
		return frame[270:460, 770:1130]
	elif speaker == "dl" or speaker == "eh":
		return  frame[int(frame.shape[0]*3/4):, int(frame.shape[1]*3/4): ]


def main():
    data_dir = Path("~/Lip2Wav/Dataset_fps25").expanduser()
    speaker_list = ["chem", "chess", "dl", "eh", "hs"]
    speaker_list = ["chem"]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)

    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        spk_dir = data_dir / speaker / "intervals"
        data_path_list = list(spk_dir.glob("*/*.mp4"))
        
        for data_path in tqdm(data_path_list):
            save_path_landmark = Path(str(data_path).replace("Dataset", "landmark").replace(".mp4", ".csv"))
            save_path_bbox = Path(str(data_path).replace("Dataset", "bbox").replace(".mp4", ".csv"))
            
            if save_path_landmark.exists() and save_path_bbox.exists():
                continue
            
            save_path_landmark.parents[0].mkdir(parents=True, exist_ok=True)
            save_path_bbox.parents[0].mkdir(parents=True, exist_ok=True)
            
            landmark_list = []
            bbox_list = []
            
            print(save_path_landmark.parents[0].name, save_path_landmark.stem)
            
            # try:
            container = av.open(str(data_path))
            stream = container.streams.video[0]
            print(stream.frames, stream.rate, stream.width, stream.height)
            
            for frame in container.decode(video=0):
                img = frame.to_image()
                arr = np.asarray(img)   # (H, W, C)
                print("crop frame")
                arr = crop_frame(arr, speaker)
                
                print("detect face")
                landmarks, landmark_scores, bboxes = fa.get_landmarks(arr, return_bboxes=True, return_landmark_score=True)
                
                max_mean = 0
                max_score_idx = 0
                for i, score in enumerate(landmark_scores):
                    score_mean = np.mean(score)
                    if score_mean > max_mean:
                        max_mean = score_mean
                        max_score_idx = i

                landmark = landmarks[max_score_idx]
                bbox = bboxes[max_score_idx][:-1]

                coords_list = []
                for coords in landmark:
                    coords_list.append(coords[0])
                    coords_list.append(coords[1])

                landmark_list.append(coords_list)
                bbox_list.append(bbox)
            
            total_frames = container.streams.video[0].frames
            print(total_frames, len(landmark_list), len(bbox_list))
            assert total_frames == len(landmark_list) 
            assert total_frames == len(bbox_list)
            
            # with open(str(save_path_landmark), "w") as f:
            #     writer = csv.writer(f)
            #     for landmark in landmark_list:
            #         writer.writerow(landmark)
                    
            # with open(str(save_path_bbox), "w") as f:
            #     writer = csv.writer(f)
            #     for bbox in bbox_list:
            #         writer.writerow(bbox)
            # except:
            #     print("error", data_path)
            #     continue                
            
        
if __name__ == "__main__":
    main()