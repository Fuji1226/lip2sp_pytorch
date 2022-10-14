import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D, Simple
from model.resnet18 import ResNet18
from model.invres import InvResNet3D
from model.mdconv import InvResNetMD
from model.transformer_remake import Encoder, OfficialEncoder
from model.conformer.encoder import ConformerEncoder
from model.nar_decoder import ResTCDecoder
from model.rnn import LSTMEncoder, GRUEncoder
from model.dilated_conv import DilatedConvEncoder


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels, norm_type,
        inv_up_scale, sq_r, md_n_groups, c_attn, s_attn,
        separate_frontend, which_res,
        d_model, n_layers, n_head, conformer_conv_kernel_size,
        rnn_hidden_channels, rnn_n_layers,
        dconv_inner_channels, dconv_kernel_size, dconv_n_layers,
        dec_n_layers, dec_inner_channels, dec_kernel_size,
        tc_n_attn_layer, tc_n_head, tc_d_model,
        feat_add_channels, feat_add_layers, 
        n_speaker, spk_emb_dim,
        which_encoder, which_decoder, apply_first_bn, use_feat_add, phoneme_classes, use_phoneme, use_dec_attention, 
        upsample_method, compress_rate,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False):
        super().__init__()

        assert d_model % n_head == 0
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.use_gc = use_gc
        self.separate_frontend = separate_frontend

        self.first_batch_norm = nn.BatchNorm3d(in_channels)

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
        elif which_res == "resnet18":
            self.ResNet_GAP = ResNet18(
                in_channels=in_channels,
                out_channels=rnn_hidden_channels,
                hidden_channels=res_inner_channels,
                dropout=res_dropout,
            )   
        elif which_res == "inv":
            self.ResNet_GAP = InvResNet3D(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
                up_scale=inv_up_scale,
                sq_r=sq_r,
                c_attn=c_attn,
                s_attn=s_attn,
            )
        elif which_res == "invmd":
            self.ResNet_GAP = InvResNetMD(
                in_channels=in_channels, 
                out_channels=rnn_hidden_channels, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type,
                up_scale=inv_up_scale,
                sq_r=sq_r,
                n_groups=md_n_groups,
                c_attn=c_attn,
                s_attn=s_attn,
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

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=rnn_hidden_channels,
            out_channels=out_channels,
            inner_channels=dec_inner_channels,
            n_layers=dec_n_layers,
            kernel_size=dec_kernel_size,
            dropout=dec_dropout,
            feat_add_channels=feat_add_channels, 
            feat_add_layers=feat_add_layers,
            use_feat_add=use_feat_add,
            phoneme_classes=phoneme_classes,
            use_phoneme=use_phoneme,
            spk_emb_dim=spk_emb_dim,
            n_attn_layer=tc_n_attn_layer,
            n_head=tc_n_head,
            d_model=tc_d_model,
            reduction_factor=reduction_factor,
            use_attention=use_dec_attention,
            upsample_method=upsample_method,
            compress_rate=compress_rate,
        )

    def forward(self, lip=None, data_len=None, gc=None):
        output = feat_add_out = phoneme = None

        # resnet
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)

        # speaker embedding
        if gc is not None:
            spk_emb = self.emb_layer(gc)
        else:
            spk_emb = None

        # decoder
        output, feat_add_out, phoneme, out_upsample = self.decoder(enc_output, spk_emb, data_len)
        
        return output, feat_add_out, phoneme