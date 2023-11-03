import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import hydra
import sys
import onnxruntime
import numpy as np
import face_alignment
import av

sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

from train_nar_with_ex_avhubert_raw import make_model
from parallelwavegan.pwg_train import make_model as make_pwg


def convert_lip2sp(cfg):
    device = torch.device('cpu')
    model = make_model(cfg, device)
    lip = torch.rand(1, 1, 88, 88, 250).to(torch.float32)
    lip_len = torch.tensor([lip.shape[-1]]).to(torch.int64)
    spk_emb = torch.rand(1, 256).to(torch.float32)
    torch.onnx.export(
        model, 
        (lip, lip_len, spk_emb), 
        'model.onnx',
        input_names=["lip", 'lip_len', 'spk_emb'],
        dynamic_axes={
            "lip": {0: "batch_size", 4:"seq_length"},
            'lip_len': {0: 'batch_size'},
            'spk_emb': {0: 'batch_size'},
        }
    )

    lip = torch.rand(1, 1, 88, 88, 100).to(torch.float32)
    lip_len = torch.tensor([lip.shape[-1]]).to(torch.int64)
    spk_emb = torch.rand(1, 256).to(torch.float32)
    model = onnxruntime.InferenceSession("model.onnx")
    mel, _ = model.run(
        None,
        {
            'lip': lip.numpy(),
            'lip_len': lip_len.numpy(),
            'spk_emb': spk_emb.numpy(),
        }
    )


def convert_pwg(cfg):
    device = torch.device('cpu')
    pwg, disc = make_pwg(cfg, device)
    feature = torch.rand(1, 80, 100)
    noise = torch.rand(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length)
    torch.onnx.export(
        pwg,
        (noise, feature),
        'pwg.onnx',
        input_names=['noise', 'feature'],
        dynamic_axes={
            'noise': {0: 'batch_size', 2: 'seq_length'},
            'feature': {0: 'batch_size', 2: 'seq_legnth'},
        }
    )

    feature = torch.rand(1, 80, 200)
    noise = torch.rand(feature.shape[0], 1, feature.shape[-1] * cfg.model.hop_length)
    pwg = onnxruntime.InferenceSession('pwg.onnx')
    wav = pwg.run(
        None,
        {
            'noise': noise.numpy(),
            'feature': feature.numpy(),
        }
    )[0]


def convert_landmark_detector():
    pass


class LandMarkDetector(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            device=device, 
            flip_input=False
        )

    def forward(self, frame):
        '''
        frame : (H, W, C)
        '''
        landmarks, landmark_scores, bboxes = self.fa.get_landmarks(
            frame,
            return_bboxes=True,
            return_landmark_score=True
        )

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

        return coords_list, bbox


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    model = LandMarkDetector('cpu').eval()
    frame = torch.rand(256, 256, 3)
    # torch.onnx.export(
    #     model,
    #     (frame),
    #     'landmark_detector.onnx',
    #     input_names=['frame'],
    #     dynamic_axes={
    #         'frame': {0: 'height', 1: 'width'}
    #     }
    # )
    # frame = torch.rand(1024, 1024, 3)
    # model = onnxruntime.InferenceSession('landmark_detector.onnx')
    # coords_list, bbox = model.run(
    #     None,
    #     {
    #         'frame': frame.numpy(),
    #     }
    # )

    # breakpoint()

    path = '/home/minami/dataset/lip/cropped_fps25/F01_kablab/BASIC5000_0278.mp4'
    container = av.open(str(path))
    for frame in container.decode(video=0):
        img = frame.to_image()
        arr = np.asarray(img)
        arr = torch.from_numpy(arr)
        model(arr)


if __name__ == '__main__':
    main()  