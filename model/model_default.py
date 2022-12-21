import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D
from model.transformer_remake import Encoder, Decoder
from model.pre_post import Postnet
from model.glu_remake import GLU
from model.rnn import LSTMEncoder, GRUEncoder


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels, which_res,
        trans_enc_n_layers, trans_enc_n_head, 
        rnn_n_layers, rnn_which_norm,
        glu_inner_channels, glu_layers, glu_kernel_size,
        trans_dec_n_layers, trans_dec_n_head,
        n_speaker, spk_emb_dim,
        pre_inner_channels, post_inner_channels, post_n_layers, post_kernel_size,
        n_position, which_encoder, which_decoder,
        dec_dropout, res_dropout, rnn_dropout, reduction_factor=2):
        super().__init__()

        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.reduction_factor = reduction_factor

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=int(res_inner_channels * 8), 
            inner_channels=res_inner_channels,
            dropout=res_dropout,
        )
        inner_channels = int(res_inner_channels * 8)
        
        # encoder
        if which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=trans_enc_n_layers, 
                n_head=trans_enc_n_head, 
                d_model=inner_channels, 
                reduction_factor=reduction_factor,  
            )
        elif which_encoder == "gru":
            self.encoder = GRUEncoder(
                hidden_channels=inner_channels,
                n_layers=rnn_n_layers,
                bidirectional=True,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)
        self.spk_emb_layer = nn.Linear(inner_channels + spk_emb_dim, inner_channels)

        # decoder
        if self.which_decoder == "glu":
            self.decoder = GLU(
                inner_channels=inner_channels, 
                out_channels=out_channels,
                pre_in_channels=int(out_channels * reduction_factor), 
                pre_inner_channels=pre_inner_channels,
                cond_channels=inner_channels,
                reduction_factor=reduction_factor, 
                n_layers=glu_layers,
                kernel_size=glu_kernel_size,
                dropout=dec_dropout,
            )
        elif self.which_decoder == "transformer":
            self.decoder = Decoder(
                dec_n_layers=trans_dec_n_layers,
                n_head=trans_dec_n_head,
                dec_d_model=inner_channels,
                pre_in_channels=int(out_channels * reduction_factor),
                pre_inner_channels=pre_inner_channels,
                out_channels=out_channels,
                n_position=n_position,
                reduction_factor=reduction_factor,
                dropout=dec_dropout,
            )

        # postnet
        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_kernel_size, post_n_layers)

    def forward(self, lip, prev=None, data_len=None, gc=None, mixing_prob=None):
        """
        lip : (B, C, H, W, T)
        prev, out, dec_output : (B, C, T)
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # resnet
        enc_output, fmaps = self.ResNet_GAP(lip)
        
        # encoder
        enc_output = self.encoder(enc_output, data_len)    # (B, T, C) 

        # speaker embedding
        if gc is not None:
            spk_emb = self.emb_layer(gc)    # (B, C)
            spk_emb = spk_emb.unsqueeze(1).expand(-1, enc_output.shape[1], -1)
            enc_output = torch.cat([enc_output, spk_emb], dim=-1)
            enc_output = self.spk_emb_layer(enc_output)
        else:
            spk_emb = None

        # decoder
        # 学習時
        if prev is not None:
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
            mixed_prev = None

        # postnet
        out = self.postnet(dec_output) 
        return out, dec_output, mixed_prev, fmaps

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
