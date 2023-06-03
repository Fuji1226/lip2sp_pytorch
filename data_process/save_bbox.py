from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import av
import time
import torchvision

# If required, create a face detection pipeline using MTCNN:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=96, margin=0, post_process=False, device=device)

speaker = "F01_kablab"
data_dir = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()
data_path_list = list(data_dir.glob("*.mp4"))

for path in tqdm(data_path_list):
    video = mmcv.VideoReader(str(path))
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    
    face_list = []
    frame_list = []
    for i, frame in enumerate(frames):
        frame = torch.from_numpy(np.array(frame))
        start = time.time()
        boxes, _ = mtcnn.detect(frame)
        end = time.time()
        
        frame_list.append(frame)
        if len(frame_list) == 50 or i == (len(frames) - 1):
            segment = torch.stack(frame_list, dim=0)
            start = time.time()
            boxes, probs, points = mtcnn.detect(segment, landmarks=True)
            end = time.time()
            frame_list.clear()
    
    # face_list = []
    # for frame in frames:
    #     frame = torch.from_numpy(np.array(frame))
    #     start = time.time()
    #     boxes, _ = mtcnn.detect(frame)
    #     end = time.time()
    #     # print(end - start)
        
    #     x = int(boxes[0][2] - boxes[0][0])
    #     y = int(boxes[0][3] - boxes[0][1])
    #     size = x if x > y else y
        
    #     x_mid = (boxes[0][2] + boxes[0][0]) // 2
    #     y_mid = (boxes[0][3] + boxes[0][1]) // 2
        
    #     frame = frame[
    #         int(x_mid - size // 2):int(x_mid + size // 2), 
    #         int(y_mid - size // 2):int(y_mid + size // 2),
    #         :
    #     ]
    #     frame = frame.permute(2, 0, 1)
    #     frame = torchvision.transforms.functional.resize(frame, [96, 96])
        
    #     # print(frame.shape, type(frame))
    #     face_list.append(frame)
        
    # face = torch.stack(face_list, dim=0)
    # face = face.permute(0, 2, 3, 1).to(torch.uint8)