from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D, DSResNet3D, DSResNet3DCbam, DSResNet3DCbamSmall, InvResNet3D
from model.transformer_remake import Encoder, OfficialEncoder
from model.conformer.encoder import ConformerEncoder
from model.audio_enc import SpeakerEncoderConv, SpeakerEncoderRNN, ContentEncoder
from model.nar_decoder import ResTCDecoder
from model.vq import VQ
from model.classifier import SpeakerClassifier
from model.grad_reversal import GradientReversal
from model.rnn import LSTMEncoder, GRUEncoder
from model.dilated_conv import DilatedConvEncoder


class LipEncoder(nn.Module):
    def __init__(
        self, in_channels, res_layers, res_inner_channels,
        d_model, n_layers, n_head, conformer_conv_kernel_size,
        res_dropout, norm_type_lip, reduction_factor, vq_num_emb,
        apply_first_bn, compress_rate, which_encoder, which_res, separate_frontend):
        super().__init__()
        assert d_model % n_head == 0
        self.apply_first_bn = apply_first_bn
        self.compress_rate = compress_rate
        self.separate_frontend = separate_frontend

        self.first_batch_norm = nn.BatchNorm3d(in_channels)

        if which_res == "default":
            self.ResNet_GAP = ResNet3D(
                in_channels=in_channels, 
                out_channels=d_model, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type_lip,
            )
        elif which_res == "ds":
            self.ResNet_GAP = DSResNet3D(
                in_channels=in_channels, 
                out_channels=d_model, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type_lip,
            )
        elif which_res == "ds_cbam":
            self.ResNet_GAP = DSResNet3DCbam(
                in_channels=in_channels, 
                out_channels=d_model, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type_lip,
            )
        elif which_res == "ds_cbam_small":
            self.ResNet_GAP = DSResNet3DCbamSmall(
                in_channels=in_channels, 
                out_channels=d_model, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type_lip,
            )
        elif which_res == "inv":
            self.ResNet_GAP = InvResNet3D(
                in_channels=in_channels, 
                out_channels=d_model, 
                inner_channels=res_inner_channels,
                layers=res_layers, 
                dropout=res_dropout,
                norm_type=norm_type_lip,
            )
        
        if which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=n_layers, 
                n_head=n_head, 
                d_model=d_model, 
                reduction_factor=reduction_factor,  
            )
        elif which_encoder == "conformer":
            self.encoder = ConformerEncoder(
                encoder_dim=d_model, 
                num_layers=n_layers, 
                num_attention_heads=n_head, 
                conv_kernel_size=conformer_conv_kernel_size,
                reduction_factor=reduction_factor,
            )
        elif which_encoder == "lstm":
            self.encoder = LSTMEncoder(
                in_channels=d_model,
                hidden_dim=d_model,
                out_channels=d_model,
                n_layers=n_layers,
                bidirectional=True,
            )
        elif which_encoder == "gru":
            self.encoder = GRUEncoder(
                in_channels=d_model,
                hidden_dim=d_model,
                out_channels=d_model,
                n_layers=n_layers,
                bidirectional=True,
            )
        elif which_encoder == "official":
            self.encoder = OfficialEncoder(
                d_model=d_model,
                nhead=n_head,
                num_layers=n_layers,
            )
        elif which_encoder == "dconv":
            self.encoder = DilatedConvEncoder(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                n_layers=5,
            )
    
        self.compress_layer_lip = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.lip2idx_layer = nn.Linear(d_model, vq_num_emb)

    def forward(self, lip, lip_delta=None, data_len=None):
        # resnet
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)
        if self.compress_rate == 4:
            enc_output = self.compress_layer_lip(enc_output.permute(0, 2, 1)).permute(0, 2, 1)
        enc_output = self.lip2idx_layer(enc_output)

        return enc_output


class VoiceConversionNetVQ(nn.Module):
    def __init__(
        self, out_channels,
        tc_d_model, tc_n_attn_layer, tc_n_head,
        dec_n_layers, dec_inner_channels, dec_kernel_size,
        feat_add_channels, feat_add_layers, 
        vq_emb_dim, vq_num_emb, spk_emb_dim, n_speaker,
        norm_type_audio,
        content_d_model, content_n_attn_layer, content_n_head, which_spk_enc, 
        use_feat_add, phoneme_classes, use_phoneme, use_dec_attention,
        upsample_method,  compress_rate,
        dec_dropout, reduction_factor=2):
        super().__init__()
        self.compress_rate = compress_rate

        if which_spk_enc == "conv":
            self.speaker_enc = SpeakerEncoderConv(
                in_channels=out_channels,
                out_channels=spk_emb_dim,
            )
        elif which_spk_enc == "rnn":
            self.speaker_enc = SpeakerEncoderRNN(
                in_channels=out_channels,
                hidden_dim=512,
                out_channels=spk_emb_dim,
                n_layers=2,
                bidirectional=True
            )
        self.content_enc = ContentEncoder(
            in_channels=out_channels,
            out_channels=content_d_model,
            n_attn_layer=content_n_attn_layer,
            n_head=content_n_head,
            reduction_factor=reduction_factor,
            norm_type=norm_type_audio,
        )

        self.compress_layer_audio = nn.Conv1d(content_d_model, content_d_model, kernel_size=3, stride=2, padding=1)

        # vq
        self.pre_vq_layer = nn.Conv1d(content_d_model, vq_emb_dim, kernel_size=1)
        self.vq = VQ(emb_dim=vq_emb_dim, num_emb=vq_num_emb)

        self.gr_layer = GradientReversal(1.0)
        self.classifier = SpeakerClassifier(vq_emb_dim, 512, n_layers=2, bidirectional=True, n_speaker=n_speaker)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=vq_emb_dim,
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

    def forward(self, feature=None, feature_ref=None, lip_enc_out=None, data_len=None):
        output = feat_add_out = phoneme = spk_emb = quantize = embed_idx = vq_loss = enc_output = idx_pred = spk_class = None

        if lip_enc_out is None:
            # speaker embedding
            spk_emb = self.speaker_enc(feature_ref)     # (B, C)
            
            # content encoder
            enc_output = self.content_enc(feature, data_len)     # (B, T, C)

            # vq
            enc_output = enc_output.permute(0, 2, 1)    # (B, C, T)
            if self.compress_rate == 4:
                enc_output = self.compress_layer_audio(enc_output)
            enc_output = self.pre_vq_layer(enc_output)
            quantize, vq_loss, embed_idx = self.vq(enc_output)
            quantize = quantize.permute(0, 2, 1)    # (B, T, C) 

            # speaker classifier
            spk_class = self.classifier(self.gr_layer(quantize))  
        
        else:
            # speaker embedding
            with torch.no_grad():
                spk_emb = self.speaker_enc(feature_ref)     # (B, C)

            idx_pred = torch.softmax(lip_enc_out, dim=-1)
            idx_pred = torch.argmax(idx_pred, dim=-1)   # (B, T)
            quantize = F.embedding(idx_pred, self.vq.embed.transpose(0, 1))     # (B, T, vq_emb_dim)

        # decoder
        if lip_enc_out is None:
            output, feat_add_out, phoneme, out_upsample = self.decoder(quantize, spk_emb, data_len)
        else:
            with torch.no_grad():
                output, feat_add_out, phoneme, out_upsample = self.decoder(quantize, spk_emb, data_len)

        return output, feat_add_out, phoneme, spk_emb, quantize, embed_idx, vq_loss, enc_output, idx_pred, spk_class, out_upsample