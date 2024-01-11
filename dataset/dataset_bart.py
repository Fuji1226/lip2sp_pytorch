import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetBART(Dataset):
    def __init__(
        self,
    ):
        super().__init__()
        
    def __len__(self):
        return
        
    def __getitem__(
        self,
    ):
        return
    
    
class TransformBART:
    def __init__(self):
        pass
    
    def __call__(self):
        return