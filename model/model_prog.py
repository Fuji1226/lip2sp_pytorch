"""
ProgressiveGANの学習方法でやるときのモデル
"""
import os
import sys
import glob

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder, Decoder
    from model.pre_post import ProgressivePostnet
    from model.conformer.encoder import Conformer_Encoder
    from model.glu_prog import ProgressiveGLU
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, Decoder
    from .pre_post import ProgressivePostnet
    from .conformer.encoder import Conformer_Encoder
    from .glu_prog import ProgressiveGLU


class ProgressiveLip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model,
        glu_inner_channels, glu_layers, glu_kernel_size,
        pre_inner_channels, post_inner_channels, post_n_layers,
        n_position, max_len, which_encoder, which_decoder, apply_first_bn,
        dropout=0.1, reduction_factor=2, use_gc=False, input_layer_dropout=False, diag_mask=False):
        super().__init__()
        assert d_model % n_head == 0
        assert which_encoder == "transformer" or "conformer"
        assert which_decoder == "transformer" or "glu"
        self.max_len = max_len
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        if apply_first_bn:
            self.first_batch_norm = nn.BatchNorm3d(in_channels)

        self.ResNet_GAP = ResNet3D(in_channels, d_model, res_layers, input_layer_dropout, dropout)

        # encoder
        if self.which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=n_layers, 
                n_head=n_head, 
                d_model=d_model, 
                n_position=n_position, 
                reduction_factor=reduction_factor, 
                dropout=0.1,    
            )
        elif self.which_encoder == "conformer":
            self.encoder = Conformer_Encoder(
                encoder_dim=d_model, 
                num_layers=n_layers, 
                num_attention_heads=n_head, 
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
                dropout=0.1,   
                use_gc=use_gc,
                diag_mask=diag_mask,
            )
        elif self.which_decoder == "glu":
            self.decoder = ProgressiveGLU(
                inner_channels=glu_inner_channels, 
                out_channels=out_channels,
                pre_in_channels=out_channels * reduction_factor, 
                pre_inner_channels=pre_inner_channels,
                reduction_factor=reduction_factor, 
                n_layers=glu_layers,
                kernel_size=glu_kernel_size,
                dropout=dropout,
            )

        self.postnet = ProgressivePostnet(
            in_channels=out_channels, 
            inner_channels=post_inner_channels, 
            out_channels=out_channels, 
            n_layers=post_n_layers,
        )

    def forward(self, lip, prev=None, data_len=None, gc=None, training_method=None, mixing_prob=None):
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature, data_len, self.max_len)    # (B, T, C)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature, data_len, self.max_len)    # (B, T, C) 

        # decoder
        # 学習時
        if prev is not None:
            if training_method == "tf":
                dec_output = self.decoder_forward(enc_output, prev, data_len, self.max_len)

            elif training_method == "ss":
                with torch.no_grad():
                    dec_output = self.decoder_forward(enc_output, prev, data_len, self.max_len)

                    # mixing_prob分だけtargetを選択し，それ以外をdec_outputに変更することで混ぜる
                    mixing_prob = torch.zeros_like(prev) + mixing_prob
                    judge = torch.bernoulli(mixing_prob)
                    mixed_prev = torch.where(judge == 1, prev, dec_output)

                # 混ぜたやつでもう一回計算させる
                dec_output = self.decoder_forward(enc_output, mixed_prev, data_len, self.max_len)
        # 推論時
        else:
            dec_output = self.decoder_inference(enc_output)

        # postnet
        out = self.postnet(dec_output) 
        return out, dec_output, enc_output

    def decoder_forward(self, enc_output, prev=None, data_len=None, max_len=None, mode="training"):
        """
        学習時の処理
        """
        if self.which_decoder == "transformer":
            dec_output = self.decoder(enc_output, prev, data_len, self.max_len, mode=mode)

        elif self.which_decoder == "glu":
            dec_output = self.decoder(enc_output, prev, mode=mode)

        return dec_output

    def decoder_inference(self, enc_output, mode="inference"):
        """
        推論時の処理
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
