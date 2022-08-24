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
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder, Decoder
    from model.pre_post import Postnet
    from model.conformer.encoder import ConformerEncoder
    from model.glu_remake import GLU
    from model.nar_decoder import FeadAddPredicter
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, Decoder
    from .pre_post import Postnet
    from .conformer.encoder import ConformerEncoder
    from .glu_remake import GLU
    from .nar_decoder import FeadAddPredicter


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model, conformer_conv_kernel_size,
        glu_inner_channels, glu_layers, glu_kernel_size,
        feat_add_channels, feat_add_layers,
        pre_inner_channels, post_inner_channels, post_n_layers,
        n_position, which_encoder, which_decoder, apply_first_bn, multi_task, add_feat_add,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False):
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
        )

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

        # decoder
        if self.which_decoder == "transformer":
            self.decoder = Decoder(
                dec_n_layers=dec_n_layers, 
                n_head=n_head, 
                dec_d_model=dec_d_model, 
                pre_in_channels=out_channels * reduction_factor, 
                pre_inner_channels=pre_inner_channels, 
                out_channels=out_channels, 
                n_position=n_position, 
                reduction_factor=reduction_factor, 
                use_gc=use_gc,
            )
        elif self.which_decoder == "glu":
            self.decoder = GLU(
                inner_channels=glu_inner_channels, 
                out_channels=out_channels,
                pre_in_channels=out_channels * reduction_factor, 
                pre_inner_channels=pre_inner_channels,
                cond_channels=d_model,
                reduction_factor=reduction_factor, 
                n_layers=glu_layers,
                kernel_size=glu_kernel_size,
                dropout=dec_dropout,
            )

        # feat_add predicter
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.feat_add_layer = FeadAddPredicter(d_model, feat_add_channels, glu_kernel_size, feat_add_layers, dec_dropout)
        self.connect_layer = nn.Conv1d(feat_add_channels, d_model, kernel_size=3, stride=2, padding=1)

        # postnet
        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_n_layers)

    def forward(self, lip, prev=None, data_len=None, gc=None, training_method=None, mixing_prob=None):
        """
        lip : (B, C, H, W, T)
        prev, out, dec_output : (B, C, T)
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 

        # feat_add predicter
        if self.multi_task:
            feat_add_out = self.upsample_layer(enc_output.permute(0, 2, 1))     # (B, C, T)
            feat_add_out = self.feat_add_layer(feat_add_out)
            if self.add_feat_add:
                feat_add_cond = self.connect_layer(feat_add_out)
                enc_output = enc_output + feat_add_cond.permute(0, 2, 1)    # (B, T, C)
        else:
            feat_add_out = None

        # decoder
        # 学習時
        if prev is not None:
            if training_method == "tf":
                dec_output = self.decoder_forward(enc_output, prev, data_len)

            elif training_method == "ss":
                with torch.no_grad():
                    dec_output = self.decoder_forward(enc_output, prev, data_len)

                    # mixing_prob分だけtargetを選択し，それ以外をdec_outputに変更することで混ぜる
                    mixing_prob = torch.zeros_like(prev) + mixing_prob
                    judge = torch.bernoulli(mixing_prob)
                    mixed_prev = torch.where(judge == 1, prev, dec_output)

                # 混ぜたやつでもう一回計算させる
                dec_output = self.decoder_forward(enc_output, mixed_prev, data_len)
        # 推論時
        else:
            dec_output = self.decoder_inference(enc_output)

        # postnet
        out = self.postnet(dec_output) 
        return out, dec_output, feat_add_out

    def decoder_forward(self, enc_output, prev=None, data_len=None, mode="training"):
        """
        学習時の処理
        enc_output : (B, T, C)
        dec_output : (B, C, T)
        """
        if self.which_decoder == "transformer":
            dec_output = self.decoder(enc_output, prev, data_len, mode=mode)

        elif self.which_decoder == "glu":
            dec_output = self.decoder(enc_output, prev, mode=mode)

        return dec_output

    def decoder_inference(self, enc_output, mode="inference"):
        """
        推論時の処理
        enc_output : (B, T, C)
        dec_output : (B, C, T)
        """
        dec_outputs = []
        max_decoder_time_steps = enc_output.shape[1] 

        if self.which_decoder == "transformer":
            for t in range(max_decoder_time_steps):
                if t == 0:
                    dec_output = self.decoder(enc_output, mode=mode)
                else:
                    dec_output = self.decoder(enc_output, dec_outputs[-1], mode=mode)
                dec_outputs.append(dec_output)

        elif self.which_decoder == "glu":
            for t in range(max_decoder_time_steps):
                if t == 0:
                    dec_output = self.decoder(enc_output[:, t, :].unsqueeze(1), mode=mode)
                else:
                    dec_output = self.decoder(enc_output[:, t, :].unsqueeze(1), dec_outputs[-1], mode=mode)
                dec_outputs.append(dec_output)

        # 溜め込んだ出力を時間方向に結合して最終出力にする
        dec_output = torch.cat(dec_outputs, dim=-1)
        assert dec_output.shape[-1] == max_decoder_time_steps * self.reduction_factor
        return dec_output

    def reset_state(self):
        self.decoder.reset_state()
