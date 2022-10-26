import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D, Simple, Simple_NonRes, SimpleBig
from model.transformer_remake import Encoder, Decoder, OfficialEncoder
from model.vq import VQ
from model.pre_post import Postnet
from model.conformer.encoder import ConformerEncoder
from model.glu_remake import GLU
from model.rnn import LSTMEncoder, GRUEncoder


class Lip2SP_VQ(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels, norm_type,
        separate_frontend, which_res,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model, conformer_conv_kernel_size,
        rnn_hidden_channels, rnn_n_layers,
        vq_emb_dim, vq_num_emb,
        glu_inner_channels, glu_layers, glu_kernel_size,
        n_speaker, spk_emb_dim,
        pre_inner_channels, post_inner_channels, post_n_layers, post_kernel_size,
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
        self.separate_frontend = separate_frontend

        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
            )
        elif which_res == "simple":
            self.ResNet_GAP = Simple(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
            )
        elif which_res == "simple_nonres":
            self.ResNet_GAP = Simple_NonRes(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
            )
        elif which_res == "simplebig":
            self.ResNet_GAP = SimpleBig(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
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
        elif which_encoder == "official":
            self.encoder = OfficialEncoder(
                d_model=d_model,
                nhead=n_head,
                num_layers=n_layers,
            )
        elif which_encoder == "lstm":
            self.encoder = LSTMEncoder(
                hidden_channels=rnn_hidden_channels,
                n_layers=rnn_n_layers,
                bidirectional=True,
                dropout=res_dropout,
                reduction_factor=reduction_factor,
            )
        elif which_encoder == "gru":
            self.encoder = GRUEncoder(
                hidden_channels=rnn_hidden_channels,
                n_layers=rnn_n_layers,
                bidirectional=True,
                dropout=res_dropout,
                reduction_factor=reduction_factor,
            )

        # vq
        self.pre_vq_layer = nn.Conv1d(rnn_hidden_channels, vq_emb_dim, kernel_size=1)
        self.vq = VQ(emb_dim=vq_emb_dim, num_emb=vq_num_emb)

        # speaker embedding
        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)
        self.spk_emb_layer = nn.Linear(vq_emb_dim + spk_emb_dim, vq_emb_dim)

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
                cond_channels=vq_emb_dim,
                reduction_factor=reduction_factor, 
                n_layers=glu_layers,
                kernel_size=glu_kernel_size,
                dropout=dec_dropout,
            )

        # postnet
        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_kernel_size, post_n_layers)

    def forward(self, lip, lip_d=None, lip_dd=None, prev=None, data_len=None, gc=None, training_method=None, mixing_prob=None):
        """
        lip : (B, C, H, W, T)
        prev, out, dec_output : (B, C, T)
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # resnet
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 
        enc_output = self.pre_vq_layer(enc_output.permute(0, 2, 1))     # (B, C, T)
        quantize, vq_loss, embed_idx = self.vq(enc_output)
        quantize = quantize.permute(0, 2, 1)    # (B, T, C) 

        # speaker embedding
        if gc is not None:
            spk_emb = self.emb_layer(gc)    # (B, C)
            spk_emb = spk_emb.unsqueeze(1).expand(-1, quantize.shape[1], -1)
            spk_emb = torch.cat([quantize, spk_emb], dim=-1)
            spk_emb = self.spk_emb_layer(spk_emb)
        else:
            spk_emb = None

        # decoder
        # 学習時
        if prev is not None:
            if training_method == "tf":
                dec_output = self.decoder_forward(quantize, prev, data_len)

            elif training_method == "ss":
                with torch.no_grad():
                    dec_output = self.decoder_forward(quantize, prev, data_len)

                    # mixing_prob分だけtargetを選択し，それ以外をdec_outputに変更することで混ぜる
                    mixing_prob = torch.zeros_like(prev) + mixing_prob
                    judge = torch.bernoulli(mixing_prob)
                    mixed_prev = torch.where(judge == 1, prev, dec_output)

                # 混ぜたやつでもう一回計算させる
                dec_output = self.decoder_forward(quantize, mixed_prev, data_len)
        # 推論時
        else:
            dec_output = self.decoder_inference(quantize)

        # postnet
        out = self.postnet(dec_output) 
        return out, dec_output, vq_loss

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
