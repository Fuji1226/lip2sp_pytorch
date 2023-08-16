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
from conformer.encoder import ConformerEncoder


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
        inner_channels = hidden_channels // 2

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

        if hasattr(self, 'encoder'):
            feature_len = torch.div(feature_len, self.reduction_factor)
            feature = self.encoder(feature, feature_len)    # (B, T, C)
        else:
            feature = feature.permute(0, 2, 1)  # (B, T, C)

        output = self.out_layer(feature)
        return output
    

class TransposedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
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
        in_channels,
        hidden_channels,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
        conf_n_layers,
        conf_n_head,
        conf_feedforward_expansion_factor,
        dec_conv_n_layers,
        dec_conv_kernel_size,
        dec_conv_dropout,
        out_channels,
    ):
        super().__init__()
        self.concat_layer = nn.Linear(in_channels, hidden_channels)

        if which_decoder == 'gru':
            self.temporal_decoder = GRUEncoder(
                hidden_channels=hidden_channels,
                n_layers=rnn_n_layers,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
        elif which_decoder == 'conformer':
            self.temporal_decoder = ConformerEncoder(
                encoder_dim=hidden_channels,
                num_layers=conf_n_layers,
                num_attention_heads=conf_n_head,
                feed_forward_expansion_factor=conf_feedforward_expansion_factor,
            )

        self.conv_decoder = ResTCDecoder(
            cond_channels=hidden_channels,
            out_channels=out_channels,
            inner_channels=hidden_channels,
            n_layers=dec_conv_n_layers,
            kernel_size=dec_conv_kernel_size,
            dropout=dec_conv_dropout,
            reduction_factor=reduction_factor,
        )

    def forward(self, feature, data_len, spk_emb, lang_id):
        '''
        feature : (B, T, C)
        data_len : (B,)
        spk_emb : (B, C)
        lang_id : (B,)
        '''
        spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)
        lang_id = lang_id.unsqueeze(1).unsqueeze(1).expand(-1, feature.shape[1], -1)
        feature = torch.cat([feature, spk_emb, lang_id], dim=-1)    # (B, T, C)
        feature = self.concat_layer(feature)
        feature = feature.permute(0, 2, 1)    # (B, C, T)
        output = self.temporal_decoder(feature, data_len)     # (B, T, C)
        output = self.conv_decoder(output)     # (B, C, T)
        return output
    

class DomainClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_conv_layers,
        conv_dropout,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
    ):
        super().__init__()
        self.first_layer = nn.Linear(in_channels, hidden_channels)

        convs = []
        for i in range(n_conv_layers):
            convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(conv_dropout),
                )
            )
        self.convs = nn.ModuleList(convs)

        self.gru = GRUEncoder(
            hidden_channels=hidden_channels,
            n_layers=rnn_n_layers,
            dropout=rnn_dropout,
            reduction_factor=reduction_factor,
            which_norm=rnn_which_norm,
        ) 

        self.out_layer = nn.Linear(hidden_channels, 1)

    def forward(self, audio_enc_output, lip_enc_output, data_len):
        '''
        audio_enc_output : (B, T, C)
        lip_enc_output : (B, T, C)
        data_len : (B,)
        '''
        output = torch.cat([audio_enc_output, lip_enc_output], dim=-1)
        output = self.first_layer(output).permute(0, 2, 1)  # (B, C, T)

        for layer in self.convs:
            output = layer(output)

        output = self.gru(output, data_len)     # (B, T, C)

        # パディング部分は平均の計算に含まない
        output_list = []
        for i in range(output.shape[0]):
            x = output[i, :data_len[i], :]  # (T, C)
            output_list.append(torch.mean(x, dim=0))
        
        output = torch.stack(output_list, dim=0)    # (B, C)
        output = self.out_layer(output)     # (B, 1)
        return output