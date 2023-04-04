import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D, NormalConv
from model.transformer_remake import Encoder, Decoder
from model.pre_post import Postnet
from model.glu_remake import GLU
from model.rnn import LSTMEncoder, GRUEncoder
from model.grad_reversal import GradientReversal
from model.classifier import SpeakerClassifier


class F0Predicter(nn.Module):
    def __init__(self, in_channels, kernel_size, n_layers, reduction_factor, dropout):
        super().__init__()
        self.reduction_factor = reduction_factor
        padding = (kernel_size - 1) // 2
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=padding)
        self.f0_convert_layer = nn.Sequential(
            nn.Conv1d(1, in_channels, kernel_size=kernel_size, stride=reduction_factor, padding=padding),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )

    def forward(self, enc_output, f0_target=None):
        """
        enc_output : (B, T, C)
        f0, f0_target : (B, 1, T)
        """
        enc_output = enc_output.permute(0, 2, 1)    # (B, C, T)

        f0 = enc_output.clone()
        f0 = F.interpolate(f0, scale_factor=self.reduction_factor, mode="nearest")
        f0 = self.dropout(f0)
        f0 = self.out_layer(f0)

        if f0_target is None:
            f0_feature = self.f0_convert_layer(f0)
        else:
            f0_feature = self.f0_convert_layer(f0_target)

        enc_output = enc_output + f0_feature
        enc_output = enc_output.permute(0, 2, 1)    # (B, T, C)
        return enc_output, f0


class F0Predicter2(nn.Module):
    def __init__(
        self, in_channels, inner_channels, rnn_n_layers, reduction_factor, dropout, rnn_which_norm,
        trans_enc_n_layers, trans_enc_n_head, which_encoder):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.conv3d = nn.ModuleList([
            NormalConv(in_channels, inner_channels, stride=2),
            nn.Dropout(dropout),
            NormalConv(inner_channels, inner_channels * 2, stride=2),
            nn.Dropout(dropout),
            NormalConv(inner_channels * 2, inner_channels * 4, stride=2),
            nn.Dropout(dropout),
            NormalConv(inner_channels * 4, inner_channels * 8, stride=2),
            nn.Dropout(dropout),
        ])
        if which_encoder == "gru":
            self.encoder = GRUEncoder(
                hidden_channels=inner_channels * 8,
                n_layers=rnn_n_layers,
                dropout=dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
        elif which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=trans_enc_n_layers, 
                n_head=trans_enc_n_head, 
                d_model=inner_channels * 8, 
                reduction_factor=reduction_factor,  
            )
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Conv1d(inner_channels * 8, 1, kernel_size=3, padding=1)

    def forward(self, lip, data_len):
        """
        lip : (B, C, H, W, T)
        data_len : (B,)
        """
        lip = lip.permute(0, 1, 4, 2, 3)
        f0 = lip
        for layer in self.conv3d:
            f0 = layer(f0)
        f0 = torch.mean(f0, dim=(3, 4))     # (B, C, T)
        f0 = self.encoder(f0, data_len)    # (B, T, C)
        f0 = f0.permute(0, 2, 1)    # (B, C, T)
        f0 = F.interpolate(f0, scale_factor=self.reduction_factor, mode="nearest")
        f0 = self.dropout(f0)
        f0 = self.out_layer(f0)
        return f0


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels, which_res,
        trans_enc_n_layers, trans_enc_n_head, 
        rnn_n_layers, rnn_which_norm,
        glu_layers, glu_kernel_size,
        trans_dec_n_layers, trans_dec_n_head,
        use_f0_predicter, f0_predicter_inner_channels, f0_predicter_rnn_n_layers,
        f0_predicter_trans_enc_n_layers, f0_predicter_trans_enc_n_head, f0_predicter_which_encoder,
        n_speaker, spk_emb_dim, use_spk_emb, where_spk_emb,
        pre_inner_channels, post_inner_channels, post_n_layers, post_kernel_size,
        n_position, which_encoder, which_decoder,
        dec_dropout, res_dropout, rnn_dropout, f0_predicter_dropout, reduction_factor=2):
        super().__init__()

        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.reduction_factor = reduction_factor
        self.where_spk_emb = where_spk_emb
        inner_channels = int(res_inner_channels * 8)

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=int(res_inner_channels * 8), 
            inner_channels=res_inner_channels,
            dropout=res_dropout,
        )
        
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
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )

        if use_spk_emb:
            self.gr_layer = GradientReversal(1.0)
            self.classfier = SpeakerClassifier(
                in_channels=inner_channels,
                hidden_channels=inner_channels,
                n_speaker=n_speaker,
            )
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

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
                use_spk_emb=use_spk_emb,
                spk_emb_dim=spk_emb_dim,
            )
        
        # postnet
        self.postnet = Postnet(out_channels, post_inner_channels, out_channels, post_kernel_size, post_n_layers)

    def forward(self, lip, lip_len, spk_emb=None, prev=None, mixing_prob=None):
        """
        lip : (B, C, H, W, T)
        prev, output, dec_output : (B, C, T)
        f0_target : (B, 1, T)
        spk_emb : (B, C)
        """
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # resnet
        enc_output, fmaps = self.ResNet_GAP(lip)    # (B, C, T)
        
        if self.where_spk_emb == "after_res":
            if hasattr(self, "spk_emb_layer"):
                if self.adversarial_learning:
                    classifier_out = self.classfier(self.gr_layer(enc_output)) 
                else:
                    classifier_out = None
                spk_emb = spk_emb.unsqueeze(-1).expand(enc_output.shape[0], -1, enc_output.shape[-1])   # (B, C, T)
                enc_output = torch.cat([enc_output, spk_emb], dim=1)
                enc_output = self.spk_emb_layer(enc_output)
            else:
                classifier_out = None
        
        # encoder
        enc_output = self.encoder(enc_output, lip_len)    # (B, T, C) 
        if self.where_spk_emb == "after_enc":
            if hasattr(self, "spk_emb_layer"):
                enc_output = enc_output.permute(0, 2, 1)    # (B, C, T)
                if self.adversarial_learning:
                    classifier_out = self.classfier(self.gr_layer(enc_output)) 
                else:
                    classifier_out = None
                spk_emb = spk_emb.unsqueeze(-1).expand(enc_output.shape[0], -1, enc_output.shape[-1])
                enc_output = torch.cat([enc_output, spk_emb], dim=1)
                enc_output = self.spk_emb_layer(enc_output)
                enc_output = enc_output.permute(0, 2, 1)    # (B, T, C)
            else:
                classifier_out = None

        # decoder
        if prev is not None:
            with torch.no_grad():
                dec_output = self.decoder_forward(enc_output, spk_emb, prev)

                # mixing_prob分だけtargetを選択し，それ以外をdec_outputに変更することで混ぜる
                mixing_prob = torch.zeros_like(prev) + mixing_prob
                judge = torch.bernoulli(mixing_prob)
                mixed_prev = torch.where(judge == 1, prev, dec_output)

            # 混ぜたやつでもう一回計算させる
            dec_output = self.decoder_forward(enc_output, spk_emb, mixed_prev)
        else:
            dec_output = self.decoder_inference(enc_output, spk_emb)
            mixed_prev = None

        # postnet
        output = self.postnet(dec_output) 
        return output, dec_output, mixed_prev, fmaps, classifier_out

    def decoder_forward(self, enc_output, spk_emb, prev, mode="training"):
        """
        学習時の処理
        enc_output : (B, T, C)
        dec_output : (B, C, T)
        """
        if self.which_decoder == "glu":
            dec_output = self.decoder(enc_output, spk_emb, mode, prev)

        return dec_output

    def decoder_inference(self, enc_output, spk_emb, mode="inference"):
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
                    dec_output = self.decoder(enc_output[:, t, :].unsqueeze(1), spk_emb, mode)
                else:
                    dec_output = self.decoder(enc_output[:, t, :].unsqueeze(1), spk_emb, mode, dec_outputs[-1])
                dec_outputs.append(dec_output)

        # 溜め込んだ出力を時間方向に結合して最終出力にする
        dec_output = torch.cat(dec_outputs, dim=-1)
        assert dec_output.shape[-1] == int(max_decoder_time_steps * self.reduction_factor)
        return dec_output

    def reset_state(self):
        self.decoder.reset_state()
