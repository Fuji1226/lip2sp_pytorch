"""
最終的なモデル
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from .net import ResNet3D
from .transformer import Postnet, Encoder, Decoder
from .glu import GLU


class Lip2SP(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers,
        d_model, n_layers, n_head, d_k, d_v, d_inner,
        glu_inner_channels, glu_layers, 
        pre_in_channels, pre_inner_channels, post_inner_channels,
        n_position, max_len, which_decoder, 
        training_method, num_passes=None, mixing_prob=None,
        dropout=0.1, reduction_factor=2, use_gc=False):

        super().__init__()

        self.max_len = max_len
        self.which_decoder = which_decoder
        self.training_method = training_method
        self.num_passes = num_passes
        self.mixing_prob = mixing_prob

        self.first_batch_norm = nn.BatchNorm3d(in_channels)
        self.ResNet_GAP = ResNet3D(in_channels, d_model, res_layers)

        self.transformer_encoder = Encoder(
            n_layers, n_head, d_k, d_v, d_model, d_inner, n_position, reduction_factor, dropout)

        if self.which_decoder == "transformer":
            self.decoder = Decoder(
                n_layers, n_head, d_k, d_v, d_model, d_inner, 
                pre_in_channels, pre_inner_channels, 
                out_channels, n_position, reduction_factor, dropout, use_gc)
        elif self.which_decoder == "glu":
            self.decoder = GLU(
                glu_inner_channels, out_channels,
                pre_in_channels, pre_inner_channels,
                reduction_factor, glu_layers)

        self.postnet = Postnet(out_channels, post_inner_channels, out_channels)

    def forward(self, lip=None, data_len=None, prev=None, gc=None):
        if lip is not None:
            # encoder
            lip = self.first_batch_norm(lip)
            lip_feature = self.ResNet_GAP(lip)
            enc_output = self.transformer_encoder(lip_feature, data_len, self.max_len)    # (B, T, C)
            
            # decoder
            if self.which_decoder == "transformer":
                dec_output = self.decoder(
                    enc_output, data_len, self.max_len, prev, 
                    training_method=self.training_method, 
                    num_passes=self.num_passes, 
                    mixing_prob=self.mixing_prob)
                out = self.postnet(dec_output)

            elif self.which_decoder == "glu":
                dec_output = self.decoder(enc_output, prev)
                out = self.postnet(dec_output)
        return out

    def inference(self, lip=None, data_len=None, prev=None, gc=None):
        if lip is not None:
            # encoder
            lip = self.first_batch_norm(lip)
            lip_feature = self.ResNet_GAP(lip)
            enc_output = self.transformer_encoder.inference(lip_feature, data_len)    # (B, T, C)
            
            # decoder
            if self.which_decoder == "transformer":
                dec_output = self.decoder.inference(enc_output, data_len, prev)
                self.pre = dec_output
                out = self.postnet(dec_output)
                
            # elif which_decoder == "glu":
            #     dec_output = self.glu_decoder.inference(enc_output, prev)
            #     self.pre = dec_output
            #     out = self.postnet(dec_output)
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
        dropout=0.1, n_position=frames, reduction_factor=2,
        which_decoder="transformer",
    )

    # training
    out = net(lip=lip, data_len=data_len, prev=acoustic_feature)
    loss_f = nn.MSELoss()
    loss = loss_f(out, acoustic_feature)
    print(loss)

    # inference
    # 口唇動画
    # lip_channels = 5
    # width = 48
    # height = 48
    # frames = 10
    # lip = torch.rand(batch_size, lip_channels, width, height, frames)
    # inference_out = net.inference(lip=lip, which_decoder="transformer")
    # print(inference_out.shape)
    




if __name__ == "__main__":
    main()