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
        use_spk_emb,
        spk_emb_dim,
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

        if use_spk_emb:
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)
        
        self.out_layer = nn.Linear(inner_channels, out_channels)

    def forward(self, lip, lip_len, spk_emb):
        '''
        lip : (B, C, H, W, T)
        lip_len : (B,)
        spk_emb : (B, C)
        '''
        enc_output, fmaps = self.ResNet_GAP(lip)  # (B, C, T)
        if hasattr(self, 'spk_emb_layer'):
            spk_emb = spk_emb.unsqueeze(-1).expand(enc_output.shape[0], -1, enc_output.shape[-1])   # (B, C, T)
            enc_output = torch.cat([enc_output, spk_emb], dim=1)
            enc_output = self.spk_emb_layer(enc_output)
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
        inner_channels = hidden_channels * 2

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
    

class AudioDecoder(nn.Module):
    def __init__(
        self,
        which_decoder,
        hidden_channels,
        reduction_factor,
        dec_conv_n_layers,
        dec_conv_kernel_size,
        dec_conv_dropout,
        out_channels,
    ):
        super().__init__()
        self.conv_decoder = ResTCDecoder(
            cond_channels=hidden_channels,
            out_channels=out_channels,
            inner_channels=hidden_channels,
            n_layers=dec_conv_n_layers,
            kernel_size=dec_conv_kernel_size,
            dropout=dec_conv_dropout,
            reduction_factor=reduction_factor,
        )

    def forward(self, enc_output):
        '''
        enc_output : (B, T, C)
        '''
        output = self.conv_decoder(enc_output)     # (B, C, T)
        return output
    

class DomainClassifierLinear(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers,
    ):
        super().__init__()
        self.first_layer = nn.Linear(in_channels, hidden_channels)
        
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.LeakyReLU(0.2),
                )
            )
        self.layers = nn.ModuleList(layers)

        self.out_layer = nn.Linear(hidden_channels, 1)

    def forward(self, enc_output, data_len):
        '''
        enc_output : (B, T, C)
        data_len : (B,)
        '''
        output = self.first_layer(enc_output)
        for layer in self.layers:
            output = layer(output)
        output = self.out_layer(output).permute(0, 2, 1)    # (B, C, T)
        return output
    

class FeatureConverter(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers,
        dropout,
    ):
        super().__init__()
        self.first_layer = nn.Linear(in_channels, hidden_channels)
        
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)

        self.out_layer = nn.Linear(hidden_channels, in_channels)

    def forward(
        self,
        x,
    ):
        '''
        x : (B, T, C)
        '''
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_layer(x)
        return x