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
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, Decoder
    from .pre_post import Postnet
    from .conformer.encoder import ConformerEncoder
    from .glu_remake import GLU
    from .nar_decoder import FeadAddPredicter
    from .model.taco import TacotronDecoder, ConvEncoder

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

class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels, norm_type,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model, conformer_conv_kernel_size,
        glu_inner_channels, glu_layers, glu_kernel_size,
        feat_add_channels, feat_add_layers,
        n_speaker, spk_emb_dim,
        pre_inner_channels, post_inner_channels, post_n_layers,
        n_position, which_encoder, which_decoder, apply_first_bn, multi_task, add_feat_add,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False, use_stop_token=False):
        super().__init__()

        assert d_model % n_head == 0
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.multi_task = multi_task
        self.add_feat_add = add_feat_add

        if apply_first_bn:
            self.first_batch_norm = nn.BatchNorm3d(in_channels)

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
        elif self.which_encoder == "conformer":
            self.encoder = ConformerEncoder(
                encoder_dim=d_model, 
                num_layers=n_layers, 
                num_attention_heads=n_head, 
                conv_kernel_size=conformer_conv_kernel_size,
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


        self.decoder = TacotronDecoder(
            enc_channels=256,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        # feat_add predicter
        self.ctc_output_layer = nn.Linear(256, 52)

        # postnet
        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_n_layers)

    def forward(self, lip, prev=None, data_len=None, gc=None, training_method=None, mixing_prob=None, use_stop_token=False):
        """
        lip : (B, C, H, W, T)
        prev, out, dec_output : (B, C, T)
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化

    
        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        
        #re-centering(linear projection)
        # lip_feature = self.re_centering(lip_feature.transpose(1, 2))
        # lip_feature = lip_feature.transpose(1, 2)
        
        
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 

        #ctc
        ctc_output = self.ctc_output_layer(enc_output)  # (B, T, C)
        
        feat_add_out = None

        # speaker embedding
        spk_emb = None

        # decoder
        # 学習時
        if not use_stop_token:
            if prev is not None:
                #print('prev is not None')
                dec_output, logit, att_w = self.decoder(enc_output=enc_output, text_len=data_len, feature_target=prev, training_method=training_method, mixing_prob=mixing_prob)
            else:
                dec_output, logit, att_w = self.decoder(enc_output, data_len) 

            # postnet
            out = self.postnet(dec_output) 
            
        else:
            if prev is not None:
                #print('prev is not None')
                dec_output, logit, att_w, stop_token = self.decoder(enc_output=enc_output, text_len=data_len, feature_target=prev, training_method=training_method, mixing_prob=mixing_prob, use_stop_token=use_stop_token)
            else:
                dec_output, logit, att_w, stop_token = self.decoder(enc_output, data_len, use_stop_token=use_stop_token) 

            # postnet
            out = self.postnet(dec_output)

        output_dict = {}
        output_dict['output'] = out
        output_dict['dec_output'] = dec_output
        output_dict['logit'] = logit
        output_dict['att_w'] = att_w
        output_dict['ctc_output'] = ctc_output
        return output_dict

 

    def reset_state(self):
        self.decoder.reset_state()

