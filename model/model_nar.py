import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D, MultiDilatedResNet3D, AttentionResNet3D
from model.nar_decoder import ResTCDecoder
from model.rnn import GRUEncoder
from model.transformer_remake import Encoder
from model.grad_reversal import GradientReversal
from model.classifier import SpeakerClassifier


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels, which_res,
        enc_channels, rnn_n_layers, trans_n_layers, trans_n_head,
        dec_n_layers, dec_inner_channels, dec_kernel_size,
        n_speaker, spk_emb_dim,
        which_encoder, which_decoder, where_spk_emb,
        dec_dropout, res_dropout, rnn_dropout, reduction_factor=2):
        super().__init__()
        self.where_spk_emb = where_spk_emb

        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=int(res_inner_channels * 8), 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
            )
        elif which_res == "md":
            self.ResNet_GAP = MultiDilatedResNet3D(
                in_channels=in_channels, 
                out_channels=int(res_inner_channels * 8), 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
                n_groups=4,
            )
        elif which_res == "atten":
            self.ResNet_GAP = AttentionResNet3D(
                in_channels=in_channels, 
                out_channels=int(res_inner_channels * 8), 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
                reduction_factor=reduction_factor,
            )
        
        if which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=trans_n_layers, 
                n_head=trans_n_head, 
                d_model=int(res_inner_channels * 8), 
                reduction_factor=reduction_factor,  
            )
        elif which_encoder == "gru":
            self.encoder = GRUEncoder(
                hidden_channels=int(res_inner_channels * 8),
                n_layers=rnn_n_layers,
                bidirectional=True,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
            )

        self.gr_layer = GradientReversal(1.0)
        self.classfier = SpeakerClassifier(
            in_channels=int(res_inner_channels * 8),
            hidden_channels=int(res_inner_channels * 8),
            n_speaker=n_speaker,
        )

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)
        self.spk_emb_layer = nn.Conv1d(int(res_inner_channels * 8) + spk_emb_dim, int(res_inner_channels * 8), kernel_size=1)

        self.decoder = ResTCDecoder(
            cond_channels=int(res_inner_channels * 8),
            out_channels=out_channels,
            inner_channels=int(res_inner_channels * 8),
            n_layers=dec_n_layers,
            kernel_size=dec_kernel_size,
            dropout=dec_dropout,
            reduction_factor=reduction_factor,
        )

    def forward(self, lip=None, data_len=None, gc=None):
        lip_feature = self.ResNet_GAP(lip)

        if self.where_spk_emb == "after_res":
            if gc is not None:
                classifier_out = self.classfier(self.gr_layer(lip_feature)) 
                spk_emb = self.emb_layer(gc)    # (B, C)
                spk_emb = spk_emb.unsqueeze(-1).expand(lip_feature.shape[0], -1, lip_feature.shape[-1])
                lip_feature = torch.cat([lip_feature, spk_emb], dim=1)
                lip_feature = self.spk_emb_layer(lip_feature)
            else:
                classifier_out = None
                spk_emb = None

        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)

        if self.where_spk_emb == "after_enc":
            if gc is not None:
                enc_output = enc_output.permute(0, 2, 1)    # (B, C, T)
                classifier_out = self.classfier(self.gr_layer(enc_output)) 
                spk_emb = self.emb_layer(gc)    # (B, C)
                spk_emb = spk_emb.unsqueeze(-1).expand(enc_output.shape[0], -1, enc_output.shape[-1])
                enc_output = torch.cat([enc_output, spk_emb], dim=1)
                enc_output = self.spk_emb_layer(enc_output)
                enc_output = enc_output.permute(0, 2, 1)    # (B, T, C)
            else:
                classifier_out = None
                spk_emb = None

        output = self.decoder(enc_output)

        return output, classifier_out
