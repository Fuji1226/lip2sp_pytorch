"""
最終的なモデル

encoder
    transformer
    conformer
decoder 
    transformer
    glu

decoderはgluが良さそうです
"""
import os
import sys
from pathlib import Path

# 親ディレクトリからのimport用
sys.path.append(str(Path("~/lip2sp_pytorch_all/lip2sp_920_re").expanduser()))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder, Decoder
    from model.pre_post import Postnet
    from model.conformer.encoder import ConformerEncoder
    from model.glu_remake import GLU
    from model.nar_decoder import FeadAddPredicter
    from model.taco import TacotronDecoder, ConvEncoder
    from model.phone_decoder import LMDecoder
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, Decoder
    from .pre_post import Postnet
    from .conformer.encoder import ConformerEncoder
    from .glu_remake import GLU
    from .nar_decoder import FeadAddPredicter
    from .taco import TacotronDecoder, ConvEncoder
    from .phone_decoder import LMDecoder

def spec_augment(y, time_ratio=0.1, freq_ratio=0.1):
    #breakpoint()
    x = y.to('cpu').clone().detach().numpy()
    
    nu, tau = x.shape[1:]

    for _ in range(8):
        flg1 = random.random()
        flg2 = random.random()
        f = np.random.randint(int(nu*freq_ratio))
        t = np.random.randint(int(tau*time_ratio))

        if flg1 < 0.5:
            if f > 0:
                f0 = np.random.randint(nu-f)
                x[:, f0:f0+f] = x[:, f0:f0+f].mean((-2, -1))[:, None, None]
        if flg2 < 0.5:
            if t > 0:
                t0 = np.random.randint(tau-t)
                x[..., t0:t0+t] = x[..., t0:t0+t].mean((-2, -1))[:, None, None]

        ans = torch.tensor(x, dtype=y.dtype, device=y.device)
        return ans

class LMModel(nn.Module):
    def __init__(self):
        super().__init__()


        self.lm_decoder = LMDecoder(dim_in=256, 
                               dim_hidden=128, 
                               dim_out=53, 
                               dim_att=32, 
                               att_filter_size=5, 
                               att_filter_num=16, 
                               sos_id=1, 
                               att_temperature=1.0,
                               num_layers=2)
        
        # feat_add predicter
        self.ctc_output_layer = nn.Linear(256, 53)

    def forward(self, att_c, text, input_len):
        """
        lip : (B, C, H, W, T)
        prev, out, dec_output : (B, C, T)
        """
        enc_output = att_c   # (B, T, C) 
       
        #ctc
        ctc_output = self.ctc_output_layer(enc_output)  # (B, T, C)
        
        #Lm Decoder
        lm_out = self.lm_decoder(enc_output,
                                input_len,
                                text)

        output_dict = {}
        output_dict['ctc_output'] = ctc_output
        output_dict['lm_output'] = lm_out
        return output_dict

 

    def reset_state(self):
        self.decoder.reset_state()

