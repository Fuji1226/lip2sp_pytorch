"""
最終的なモデル
"""

import torch
import torch.nn as nn
from wavenet.submodules.mychainerutils.hparam_tf import HParams
from net import ResNet3D
from transformer import Prenet, Postnet, Encoder, Decoder
from glu import GLU

import os
os.environ['PYTHONBREAKPOINT'] = ''


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers,
        d_model, n_layers, n_head, d_k, d_v, d_inner,
        glu_inner_channels, glu_layers,
        pre_in_channels, pre_inner_channels, post_inner_channels,
        dropout=0.1, n_position=150, reduction_factor=2, use_gc=False):

        super().__init__()

        self.first_batch_norm = nn.BatchNorm3d(in_channels)
        self.ResNet_GAP = ResNet3D(in_channels, d_model, res_layers)

        self.transformer_encoder = Encoder(
            n_layers, n_head, d_k, d_v, d_model, d_inner, dropout, n_position)

        self.transformer_decoder = Decoder(
            n_layers, n_head, d_k, d_v, d_model, d_inner, 
            pre_in_channels, pre_inner_channels, 
            out_channels, use_gc, dropout, n_position, reduction_factor)

        self.glu_decoder = GLU(
            glu_inner_channels, out_channels,
            pre_in_channels, pre_inner_channels,
            reduction_factor, glu_layers)

        self.postnet = Postnet(out_channels, post_inner_channels, out_channels)

    def forward(self, lip=None, data_len=None, prev=None, gc=None, which_decoder=None):
        if lip is not None:
            # encoder
            lip = self.first_batch_norm(lip)
            lip_feature = self.ResNet_GAP(lip)
            enc_output = self.transformer_encoder(lip_feature, data_len)    # (B, T, C)
            breakpoint()
            
            # decoder
            if which_decoder == "transformer":
                dec_output = self.transformer_decoder(enc_output, data_len, prev)
                breakpoint()
                self.pre = dec_output
                out = self.postnet(dec_output)
                breakpoint()

            elif which_decoder == "glu":
                dec_output = self.glu_decoder(enc_output, prev)
                breakpoint()
                self.pre = dec_output
                out = self.postnet(dec_output)
                breakpoint()

        return out


def main():
    batch_size = 8

    # data_len
    data_len = [300, 300, 300, 300, 100, 100, 200, 200]
    data_len = torch.tensor(data_len)

    # 口唇動画
    lip_channels = 5
    width = 48
    height = 48
    frames = 150
    lip = torch.rand(batch_size, lip_channels, width, height, frames)

    # 音響特徴量
    feature_channels = 80
    acoustic_feature = torch.rand(batch_size, feature_channels, frames * 2)

    # parameter
    in_channels = lip_channels
    out_channels = feature_channels
    res_layers = 5
    d_model = 256
    n_layers = 2
    n_head = 8
    d_k = d_model // n_head
    d_v = d_model // n_head
    d_inner = 512
    glu_inner_channels = 256
    glu_layers = 4
    pre_in_channels = feature_channels * 2
    pre_inner_channels = 32
    post_inner_channels = 512

    # build
    net = Lip2SP(
        in_channels, out_channels, res_layers,
        d_model, n_layers, n_head, d_k, d_v, d_inner,
        glu_inner_channels, glu_layers,
        pre_in_channels, pre_inner_channels, post_inner_channels,
        dropout=0.1, n_position=frames, reduction_factor=2
    )

    out = net(lip=lip, data_len=data_len, prev=acoustic_feature, which_decoder="glu")



if __name__ == "__main__":
    main()