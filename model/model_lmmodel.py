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

class Lip2LM(nn.Module):
    def __init__(
        self, in_channels=3, res_layers=3, res_inner_channels=128, norm_type='bn',
        d_model=256, n_layers=1, n_head=4,
        which_encoder='transformer',
        res_dropout=0.5, reduction_factor=2):
        super().__init__()

        assert d_model % n_head == 0
        self.which_encoder = which_encoder
        
        self.reduction_factor = reduction_factor
        self.out_channels = 52

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=d_model, 
            inner_channels=res_inner_channels,
            layers=res_layers, 
            dropout=res_dropout,
            norm_type=norm_type,
        )

        #re-centering(linear projection)
        self.re_centering = nn.Linear(d_model, d_model)
    
        # encoder
        if self.which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=n_layers, 
                n_head=n_head, 
                d_model=d_model, 
                reduction_factor=reduction_factor,  
            )
        elif self.which_encoder == "convencoder":  
            self.encoder = ConvEncoder(
                #embed_dim=d_model,  # 文字埋め込みの次元数
                hidden_channels=d_model,  # 隠れ層の次元数
                conv_n_layers=3,  # 畳み込み層数
                #conv_channels=d_model,  # 畳み込み層のチャネル数
                conv_kernel_size=7,  # 畳み込み層のカーネルサイズ
                rnn_n_layers=1,
                dropout=0.5,  # Dropout 率
            )

        
        # feat_add predicter
        self.ctc_output_layer = nn.Linear(256, 53)
        
        # データの次元数を設定
        dim_in = 256
        dim_hidden = 128
        dim_out = 53
        dim_att = 32
        att_filter_size = 5
        att_filter_num = 16
        sos_id = 1
        att_temperature = 1.0

        self.decoder = LMDecoder(dim_in=dim_in, 
                               dim_hidden=dim_hidden, 
                               dim_out=dim_out, 
                               dim_att=dim_att, 
                               att_filter_size=att_filter_size, 
                               att_filter_num=att_filter_num, 
                               sos_id=sos_id, 
                               att_temperature=att_temperature,
                               num_layers=2)



    def forward(self, lip, prev=None, data_len=None, input_len=None):
        """
        lip : (B, C, H, W, T)
        prev, out, dec_output : (B, C, T)
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化


        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        
        #re-centering(linear projection)
        # lip_feature = self.re_centering(lip_feature.transpose(1, 2))
        # lip_feature = lip_feature.transpose(1, 2)
        
        
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 

        # デコーダに入力する
        dec_out = self.decoder(enc_output,
                               input_len,
                               prev)

        output_dict = {}
        output_dict['dec_output'] = dec_out
        return output_dict
 

    def reset_state(self):
        self.decoder.reset_state()

