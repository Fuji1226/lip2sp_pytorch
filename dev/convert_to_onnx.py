import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import hydra
import sys
import onnxruntime
import numpy as np

sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

from train_nar_with_ex_avhubert_raw import make_model


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
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


if __name__ == '__main__':
    main()