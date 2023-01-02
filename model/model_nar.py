import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from net import ResNet3D
from nar_decoder import ResTCDecoder
from rnn import GRUEncoder
from transformer_remake import Encoder
from grad_reversal import GradientReversal
from classifier import SpeakerClassifier
from landmark_net import LandmarkEncoder


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels, which_res,
        rnn_n_layers, rnn_which_norm, trans_n_layers, trans_n_head,
        use_landmark, lm_enc_inner_channels, lmco_kernel_size, lmco_n_layers, 
        lm_enc_compress_time_axis, astt_gcn_n_layers, astt_gcn_n_head, lm_enc_n_nodes,
        dec_n_layers, dec_kernel_size,
        n_speaker, spk_emb_dim,
        which_encoder, which_decoder, where_spk_emb,
        dec_dropout, res_dropout, lm_enc_dropout, rnn_dropout, reduction_factor=2):
        super().__init__()
        self.where_spk_emb = where_spk_emb

        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=int(res_inner_channels * 8), 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
            )
            inner_channels = int(res_inner_channels * 8)

        if use_landmark:
            self.landmark_encoder = LandmarkEncoder(
                inner_channels=lm_enc_inner_channels,
                lmco_kernel_size=lmco_kernel_size,
                lmco_n_layers=lmco_n_layers,
                compress_time_axis=lm_enc_compress_time_axis,
                astt_gcn_n_layers=astt_gcn_n_layers,
                astt_gcn_n_head=astt_gcn_n_head,
                n_nodes=lm_enc_n_nodes,
                dropout=lm_enc_dropout,
            )
            self.landmark_aggregate_layer = nn.Conv1d(lm_enc_inner_channels + inner_channels, inner_channels, kernel_size=1)
        
        if which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=trans_n_layers, 
                n_head=trans_n_head, 
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

        self.gr_layer = GradientReversal(1.0)
        self.classfier = SpeakerClassifier(
            in_channels=inner_channels,
            hidden_channels=inner_channels,
            n_speaker=n_speaker,
        )

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)
        self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        self.decoder = ResTCDecoder(
            cond_channels=inner_channels,
            out_channels=out_channels,
            inner_channels=inner_channels,
            n_layers=dec_n_layers,
            kernel_size=dec_kernel_size,
            dropout=dec_dropout,
            reduction_factor=reduction_factor,
        )

    def forward(self, lip, landmark=None, data_len=None, gc=None):
        """
        lip : (B, C, H, W, T)
        landmark : (B, T, 2, 68)
        """
        enc_output, fmaps = self.ResNet_GAP(lip)  # (B, C, T)

        print(lip.shape, landmark.shape)

        if hasattr(self, "landmark_encoder"):
            landmark_feature = self.landmark_encoder(landmark)  # (B, C, T)
            enc_output = self.landmark_aggregate_layer(torch.cat([enc_output, landmark_feature], dim=1))  # (B, C, T)

        if self.where_spk_emb == "after_res":
            if gc is not None:
                classifier_out = self.classfier(self.gr_layer(enc_output)) 
                spk_emb = self.emb_layer(gc)    # (B, C)
                spk_emb = spk_emb.unsqueeze(-1).expand(enc_output.shape[0], -1, enc_output.shape[-1])
                enc_output = torch.cat([enc_output, spk_emb], dim=1)
                enc_output = self.spk_emb_layer(enc_output)
            else:
                classifier_out = None
                spk_emb = None

        enc_output = self.encoder(enc_output, data_len)    # (B, T, C)

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

        return output, classifier_out, fmaps
