import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from net import ResNet3D, ResNet3DVTP, ResNet3DRemake
from nar_decoder import ResTCDecoder, LinearDecoder
from rnn import GRUEncoder
from transformer_remake import Encoder
from grad_reversal import GradientReversal
from classifier import SpeakerClassifier
from resnet18 import ResNet18
from conformer.encoder import ConformerEncoder
from avhubert import AVHuBERT


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels, which_res,
        rnn_n_layers, rnn_which_norm, trans_n_layers, trans_n_head, trans_pos_max_len,
        conf_n_layers, conf_n_head, conf_feedforward_expansion_factor,
        dec_n_layers, dec_kernel_size,
        n_speaker, spk_emb_dim,
        cfg,
        which_encoder, which_decoder, where_spk_emb, use_spk_emb,
        dec_dropout, res_dropout, rnn_dropout, is_large, adversarial_learning, reduction_factor):
        super().__init__()
        self.where_spk_emb = where_spk_emb
        self.adversarial_learning = adversarial_learning
        inner_channels = int(res_inner_channels * 8)

        #?resnetの選択
        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=inner_channels, 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
            )
        elif which_res == "default_remake":
            self.ResNet_GAP = ResNet3DRemake(
                in_channels=in_channels, 
                out_channels=inner_channels, 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
                is_large=is_large,
            )
        elif which_res == "vtp":
            self.ResNet_GAP = ResNet3DVTP(
                in_channels=in_channels, 
                out_channels=inner_channels, 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
            )
        elif which_res == "resnet18":
            self.ResNet_GAP = ResNet18(
                in_channels=in_channels,
                hidden_channels=res_inner_channels,
                dropout=res_dropout,
            )
        #!elif which_res == "avhubert" :


        #?エンコーダの選択
        if which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=trans_n_layers, 
                n_head=trans_n_head, 
                d_model=inner_channels, 
                reduction_factor=reduction_factor,  
                pos_max_len=trans_pos_max_len,
            )
        elif which_encoder == "gru":
            self.encoder = GRUEncoder(
                hidden_channels=inner_channels,
                n_layers=rnn_n_layers,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
        elif which_encoder == "conformer":
            self.encoder = ConformerEncoder(
                encoder_dim=inner_channels,
                num_layers=conf_n_layers,
                num_attention_heads=conf_n_head,
                feed_forward_expansion_factor=conf_feedforward_expansion_factor,
            )
            #!↓要確認
        elif which_encoder == "avhubert":
            self.encoder = AVHuBERT(
                cfg=cfg
            )

        #?話者特徴量
        if use_spk_emb:
            self.gr_layer = GradientReversal(1.0)
            self.classfier = SpeakerClassifier(
                in_channels=inner_channels,
                hidden_channels=inner_channels,
                n_speaker=n_speaker,
            )
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        #?デコーダの選択
        if which_decoder == 'restc':
            self.decoder = ResTCDecoder(
                cond_channels=inner_channels,
                out_channels=out_channels,
                inner_channels=inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
                reduction_factor=reduction_factor,
            )
        elif which_decoder == 'linear':
            self.decoder = LinearDecoder(
                in_channels=inner_channels,
                out_channels=out_channels,
                reduction_factor=reduction_factor,
            )

    def forward(self, lip, lip_len, spk_emb=None):
        """
        lip : (B, C, H, W, T)
        lip_len : (B,)
        spk_emb : (B, C)
        """
        enc_output, fmaps = self.ResNet_GAP(lip)  # (B, C, T)

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

        output = self.decoder(enc_output)

        return output, classifier_out, fmaps
