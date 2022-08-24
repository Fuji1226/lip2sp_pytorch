"""
Lipreadingに使用する予定のモデル
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder, PhonemeDecoder
    from model.conformer.encoder import ConformerEncoder
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, PhonemeDecoder
    from .conformer.encoder import ConformerEncoder


def token_mask(x):
    """
    音素ラベルに対してのマスクを作成
    MASK_INDEXに一致するところをマスクする

    x : (B, T)
    mask : (B, T)
    """
    MASK_INDEX = 0
    zero_matrix = torch.zeros_like(x)
    one_matrix = torch.ones_like(x)
    mask = torch.where(x == MASK_INDEX, one_matrix, zero_matrix).bool() 
    return mask


class Lip2Text(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels,
        d_model, n_layers, n_head, dec_n_layers, dec_d_model, dec_n_head, conformer_conv_kernel_size,
        which_encoder, which_decoder, apply_first_bn,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False):
        super().__init__()

        assert d_model % n_head == 0
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.n_head = n_head

        if apply_first_bn:
            self.first_batch_norm = nn.BatchNorm3d(in_channels)

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=d_model, 
            inner_channels=res_inner_channels,
            layers=res_layers, 
            dropout=res_dropout,
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

        self.connect_layer = nn.Conv1d(d_model, dec_d_model, kernel_size=1)

        # decoder
        self.decoder = PhonemeDecoder(
            dec_n_layers=dec_n_layers,
            n_head=dec_n_head,
            d_model=dec_d_model,
            out_channels=out_channels,
            reduction_factor=reduction_factor,
        )

    def forward(self, lip, prev=None, data_len=None, training_method=None, mixing_prob=None, n_max_loop=None):
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # encoder
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        if self.which_encoder == "transformer":
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)
        elif self.which_encoder == "conformer":
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C) 

        enc_output = self.connect_layer(enc_output.permute(0, -1, 1))
        enc_output = enc_output.permute(0, -1, 1)   # (B, T, C)

        # decoder
        # train
        if prev is not None:
            if training_method == "tf":
                output = self.decoder_forward(enc_output, prev, data_len)

            elif training_method == "ss":
                with torch.no_grad():
                    output = self.decoder_forward(enc_output, prev, data_len)
                    # min_prob, _ = output.min(dim=1)
                    # output[:, 0, :] = min_prob
 
                    # softmaxを適用して確率に変換
                    output = torch.softmax(output, dim=1)

                    # Onehot
                    output = torch.distributions.OneHotCategorical(output).sample()

                    # 最大値(Onehotの1のところ)のインデックスを取得
                    output = output.max(dim=1)[1]   # (B, T)
                    assert output.shape == prev.shape

                    # mixing_prob分だけラベルを選択し，それ以外を変更することで混ぜる
                    mixing_prob = torch.zeros_like(prev) + mixing_prob
                    judge = torch.bernoulli(mixing_prob)
                    mixed_prev = torch.where(judge == 1, prev, output)

                # 混ぜたやつでもう一回計算してそれを出力とする
                output = self.decoder_forward(enc_output, mixed_prev, data_len)
        # inference
        else:
            output = self.decoder_inference_greedy(enc_output, n_max_loop)

        return output

    def decoder_forward(self, enc_output, prev, data_len, mode="training"):
        """
        学習時の処理
        """
        output = self.decoder(enc_output, prev, data_len, mode=mode)
        return output

    def decoder_inference_greedy(self, enc_output, n_max_loop, mode="inference"):
        """
        greedy searchによる推論

        enc_output : (B, T, C)
        n_max_loop : eosが出なくてloopが終わらなくなる可能性があるので,上限を定めるための値
        """
        # sosとeosを表すインデックス(phoneme_encode.pyで決まる)
        SOS_INDEX = 1
        EOS_INDEX = 2

        B, T, C = enc_output.shape 

        start_phoneme_index = torch.zeros(B, 1).to(device=enc_output.device, dtype=torch.long)
        start_phoneme_index[:] = SOS_INDEX        

        outputs = []

        for t in range(n_max_loop):
            if t > 0:
                # 最初以外は前時刻の出力を入力
                prev = outputs[-1]
            else:
                # 一番最初はsosから
                prev = start_phoneme_index

            output = self.decoder(enc_output, prev, mode=mode)
            
            # softmaxを適用して確率に変換
            output = torch.softmax(output[:, :, -1].unsqueeze(-1), dim=1)

            # Onehot
            output = torch.distributions.OneHotCategorical(output).sample()

            # 最大値(Onehotの1のところ)のインデックスを取得
            output = output.max(dim=1)[1]   # (B, T)

            outputs.append(output)

            # もしeosが出たらそこで終了
            if output == EOS_INDEX:
                break
        
        # 最終出力
        output = torch.cat(outputs, dim=-1)     # (B, T)
        return output

    def decoder_inference_beam_search(self, enc_output, n_max_loop):
        """
        beam searchによる推論

        enc_output : (B, T, C)
        n_max_loop : eosが出なくてloopが終わらなくなる可能性があるので,上限を定めるための値
        """
        # sosとeosを表すインデックス(phoneme_encode.pyで指定していたもの)
        SOS_INDEX = 0
        EOS_INDEX = 1

        B, T, C = enc_output.shape 

        start_phoneme_index = torch.zeros(B, 1)
        start_phoneme_index[:] = SOS_INDEX        

        outputs = []
        return

    def reset_state(self):
        self.decoder.reset_state()