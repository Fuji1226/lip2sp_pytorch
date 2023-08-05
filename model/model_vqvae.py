
from pathlib import Path
import sys
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))
sys.path.append(str(Path('~/lip2sp_pytorch/model').expanduser()))

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
from model.vq import VQ
from conformer.encoder import ConformerEncoder
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LipEncoder(nn.Module):
    def __init__(
        self,
        which_res,
        in_channels,
        res_inner_channels,
        res_dropout,
        is_large,
        which_encoder,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
        conf_n_layers,
        conf_n_head,
        conf_feedforward_expansion_factor,
        out_channels,
    ):
        super().__init__()
        inner_channels = int(res_inner_channels * 8)

        if which_res == 'default_remake':
            self.ResNet_GAP = ResNet3DRemake(
                in_channels=in_channels, 
                out_channels=inner_channels, 
                inner_channels=res_inner_channels,
                dropout=res_dropout,
                is_large=is_large,
            )
        elif which_res == 'resnet18':
            self.ResNet_GAP = ResNet18(
                in_channels=in_channels,
                hidden_channels=res_inner_channels,
                dropout=res_dropout,
            )

        if which_encoder == 'gru':
            self.encoder = GRUEncoder(
                hidden_channels=inner_channels,
                n_layers=rnn_n_layers,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
        elif which_encoder == 'conformer':
            self.encoder = ConformerEncoder(
                encoder_dim=inner_channels,
                num_layers=conf_n_layers,
                num_attention_heads=conf_n_head,
                feed_forward_expansion_factor=conf_feedforward_expansion_factor,
            )
        
        self.out_layer = nn.Linear(inner_channels, out_channels)

    def forward(self, lip, lip_len):
        '''
        lip : (B, C, H, W, T)
        lip_len : (B,)
        '''
        enc_output, fmaps = self.ResNet_GAP(lip)  # (B, C, T)
        enc_output = self.encoder(enc_output, lip_len)    # (B, T, C)
        enc_output = self.out_layer(enc_output)
        return enc_output
    

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        conv_dropout,
        which_encoder,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
        conf_n_layers,
        conf_n_head,
        conf_feedforward_expansion_factor,
        out_channels,
    ):
        super().__init__()
        self.reduction_factor = reduction_factor
        inner_channels = int(hidden_channels * 2)

        conv_layers = [
            ConvBlock(in_channels, hidden_channels, conv_dropout),
            ConvBlock(hidden_channels, inner_channels, conv_dropout)
        ]
        self.conv_layers = nn.ModuleList(conv_layers)

        if which_encoder == 'gru':
            self.encoder = GRUEncoder(
                hidden_channels=inner_channels,
                n_layers=rnn_n_layers,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
        elif which_encoder == 'conformer':
            self.encoder = ConformerEncoder(
                encoder_dim=inner_channels,
                num_layers=conf_n_layers,
                num_attention_heads=conf_n_head,
                feed_forward_expansion_factor=conf_feedforward_expansion_factor,
            )

        self.out_layer = nn.Linear(inner_channels, out_channels)

    def forward(self, feature, feature_len):
        '''
        feature : (B, C, T)
        faeture_len : (B,)
        '''
        for layer in self.conv_layers:
            feature = layer(feature)    # (B, C, T)

        feature_len = torch.div(feature_len, self.reduction_factor)
        feature = self.encoder(feature, feature_len)    # (B, T, C)
        output = self.out_layer(feature)
        return output


class TransposedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        dropout,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class AudioDecoder(nn.Module):
    def __init__(
        self,
        which_decoder,
        hidden_channels,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
        conf_n_layers,
        conf_n_head,
        conf_feedforward_expansion_factor,
        tconv_dropout,
        out_channels,
    ):
        super().__init__()

        if which_decoder == 'gru':
            self.decoder = GRUEncoder(
                hidden_channels=hidden_channels,
                n_layers=rnn_n_layers,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
        elif which_decoder == 'conformer':
            self.decoder = ConformerEncoder(
                encoder_dim=hidden_channels,
                num_layers=conf_n_layers,
                num_attention_heads=conf_n_head,
                feed_forward_expansion_factor=conf_feedforward_expansion_factor,
            )

        tconv_layers = [
            TransposedConvBlock(hidden_channels, tconv_dropout),
            TransposedConvBlock(hidden_channels // 2, tconv_dropout)
        ]
        self.tconv_layers = nn.ModuleList(tconv_layers)

        self.out_layer = nn.Conv1d(hidden_channels // 4, out_channels, kernel_size=1)

    def forward(self, vq_feature, data_len, spk_emb, lang_id):
        '''
        vq_feature : (B, T, C)
        data_len : (B,)
        spk_emb : (B, C)
        lang_id : (B,)
        '''
        spk_emb = spk_emb.unsqueeze(1).expand(-1, vq_feature.shape[1], -1)
        lang_id = lang_id.unsqueeze(1).unsqueeze(1).expand(-1, vq_feature.shape[1], -1)
        vq_feature = torch.cat([vq_feature, spk_emb, lang_id], dim=-1)
        vq_feature = vq_feature.permute(0, 2, 1)    # (B, C, T)

        output = self.decoder(vq_feature, data_len)     # (B, T, C)
        output = output.permute(0, 2, 1)    # (B, C, T)

        for layer in self.tconv_layers:
            output = layer(output)

        output = self.out_layer(output)     # (B, C, T)
        return output
    


# import hydra
# import random
# @hydra.main(version_base=None, config_name="config", config_path="../conf")
# def main(cfg):
#     lip_encoder = LipEncoder(
#         which_res=cfg.model.which_res,
#         in_channels=cfg.model.in_channels,
#         res_inner_channels=cfg.model.res_inner_channels,
#         res_dropout=cfg.train.res_dropout,
#         is_large=cfg.model.is_large,
#         which_encoder=cfg.model.which_encoder,
#         rnn_n_layers=cfg.model.rnn_n_layers,
#         rnn_dropout=cfg.train.rnn_dropout,
#         reduction_factor=cfg.model.reduction_factor,
#         rnn_which_norm=cfg.model.rnn_which_norm,
#         conf_n_layers=cfg.model.conf_n_layers,
#         conf_n_head=cfg.model.conf_n_head,
#         conf_feedforward_expansion_factor=cfg.model.conf_feed_forward_expansion_factor,
#         out_channels=cfg.model.vq_num_emb,
#     )
#     audio_encoder = AudioEncoder(
#         in_channels=cfg.model.n_mel_channels,
#         hidden_channels=cfg.model.audio_enc_hidden_channels,
#         conv_dropout=cfg.model.audio_enc_conv_dropout,
#         which_encoder=cfg.model.audio_enc_which_encoder,
#         rnn_n_layers=cfg.model.audio_enc_rnn_n_layers,
#         rnn_dropout=cfg.model.audio_enc_rnn_dropout,
#         reduction_factor=cfg.model.reduction_factor,
#         rnn_which_norm=cfg.model.rnn_which_norm,
#         conf_n_layers=cfg.model.audio_enc_conf_n_Layers,
#         conf_n_head=cfg.model.audio_enc_conf_n_head,
#         conf_feedforward_expansion_factor=cfg.model.audio_enc_conf_feed_forward_expansion_factor,
#         out_channels=cfg.model.vq_emb_dim,
#     )
#     vq = VQ(
#         emb_dim=cfg.model.vq_emb_dim,
#         num_emb=cfg.model.vq_num_emb,
#     )
#     audio_decoder = AudioDecoder(
#         which_decoder=cfg.model.audio_dec_which_decoder,
#         hidden_channels=cfg.model.vq_emb_dim + cfg.model.spk_emb_dim + 1,
#         rnn_n_layers=cfg.model.audio_dec_rnn_n_layers,
#         rnn_dropout=cfg.model.audio_dec_rnn_dropout,
#         reduction_factor=cfg.model.reduction_factor,
#         rnn_which_norm=cfg.model.rnn_which_norm,
#         conf_n_layers=cfg.model.audio_dec_conf_n_layers,
#         conf_n_head=cfg.model.audio_dec_conf_n_head,
#         conf_feedforward_expansion_factor=cfg.model.audio_dec_conf_feedforward_expansion_factor,
#         tconv_dropout=cfg.model.audio_dec_tconv_dropout,
#         out_channels=cfg.model.n_mel_channels,
#     )

#     B = 4
#     lip = torch.rand(B, 1, 88, 88, 250)
#     feature = torch.rand(B, 80, 1000)
#     lip_len = torch.tensor([random.randint(100, 251) for _ in range(B)])
#     feature_len = torch.mul(lip_len, cfg.model.reduction_factor)

#     lip_enc_output = lip_encoder(lip, lip_len)
#     audio_enc_output = audio_encoder(feature, feature_len)

#     audio_enc_output = audio_enc_output.permute(0, 2, 1)    # (B, C, T)
#     quantize_audio, vq_loss, embed_idx = vq(audio_enc_output)
#     quantize_audio = quantize_audio.permute(0, 2, 1)
    
#     vq_idx_pred = torch.argmax(lip_enc_output, dim=-1)
#     quantize_lip = F.embedding(vq_idx_pred, vq.embed.transpose(0, 1))

#     spk_emb = torch.rand(B, 256)
#     lang_id = torch.randint(0, 2, (B,))
#     feature_pred = audio_decoder(quantize_audio, lip_len, spk_emb, lang_id)

#     loss = F.cross_entropy(lip_enc_output.permute(0, 2, 1), embed_idx)
#     print(loss)
    

# if __name__ == '__main__':
#     main()