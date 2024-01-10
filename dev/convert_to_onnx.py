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
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from torchlm.runtime import faceboxesv2_ort, pipnet_ort

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
    face_detector = faceboxesv2(device="cpu")
    face_detector.apply_exporting(
        onnx_path="./face_detector.onnx",
        opset=12, 
        simplify=True,
        output_names=None,
    )

    landmark_detector = pipnet(
        backbone="resnet101",
        pretrained=True,
        num_nb=10,
        num_lms=68,
        net_stride=32,
        input_size=256,
        meanface_type="300w",
        map_location="cpu",
        checkpoint=None,
    )
    landmark_detector.apply_exporting(
        onnx_path="./landmark_detector_resnet101.onnx",
        opset=12, 
        simplify=True,
        output_names=None,
    )


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    convert_lip2sp(cfg)
    convert_pwg(cfg)
    convert_landmark_detector()


if __name__ == '__main__':
    main()  