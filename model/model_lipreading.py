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
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder, PhonemeDecoder


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
        self, in_channels, out_channels, res_inner_channels,
        trans_enc_n_layers, trans_enc_n_head, trans_dec_n_layers, trans_dec_n_head,
        res_dropout, reduction_factor, which_encoder):
        super().__init__()
        self.out_channels = out_channels

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=int(res_inner_channels * 8), 
            inner_channels=res_inner_channels,
            dropout=res_dropout,
        )
        inner_channels = int(res_inner_channels * 8)

        # encoder
        self.encoder = Encoder(
            n_layers=trans_enc_n_layers, 
            n_head=trans_enc_n_head, 
            d_model=inner_channels, 
            reduction_factor=reduction_factor,  
        )

        # decoder
        self.decoder = PhonemeDecoder(
            dec_n_layers=trans_dec_n_layers,
            n_head=trans_dec_n_head,
            d_model=inner_channels,
            out_channels=out_channels,
            reduction_factor=reduction_factor,
        )

    def forward(self, lip, prev=None, data_len=None, training_method=None, mixing_prob=None, n_max_loop=None):
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # encoder
        enc_output, fmaps = self.ResNet_GAP(lip)
        
        enc_output = self.encoder(enc_output, data_len)    # (B, T, C)

        # decoder
        # train
        if prev is not None:
            if training_method == "tf":
                output = self.decoder_forward(enc_output, prev, data_len)

            elif training_method == "ss":
                with torch.no_grad():
                    output = self.decoder_forward(enc_output, prev, data_len)
 
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
            # output = self.decoder_inference_greedy(enc_output, n_max_loop)
            output = self.decoder_inference_beam_search(enc_output, n_max_loop)

        return output

    def decoder_forward(self, enc_output, prev, data_len, mode="training"):
        """
        学習時の処理
        """
        output = self.decoder(enc_output, prev, data_len, mode=mode)    # (B, C, T)
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
        outputs.append(start_phoneme_index)

        for t in range(n_max_loop):
            if t > 0:
                prev = torch.cat(outputs, dim=-1)
            else:
                # 一番最初はsosから
                prev = start_phoneme_index

            output = self.decoder(enc_output, prev, mode=mode)  # (B, C, T)

            # 最大値のインデックスを取得
            output = output[..., -1].unsqueeze(-1).max(dim=1)[1]   # (B, 1)

            outputs.append(output)

            # もしeosが出たらそこで終了
            if output == EOS_INDEX:
                break
        
        # 最終出力
        output = torch.cat(outputs[1:], dim=-1)     # (B, T)
        return output

    def decoder_inference_beam_search(self, enc_output, n_max_loop, beam_size=5, mode="inference"):
        """
        beam searchによる推論

        enc_output : (B, T, C)
        n_max_loop : eosが出なくてloopが終わらなくなる可能性があるので,上限を定めるための値
        """
        # sosとeosを表すインデックス(phoneme_encode.pyで指定していたもの)
        SOS_INDEX = 1
        EOS_INDEX = 2

        B, T, C = enc_output.shape 

        start_phoneme_index = torch.zeros(B, 1).to(device=enc_output.device, dtype=torch.long)
        start_phoneme_index[:] = SOS_INDEX   

        output_list = []
        n_finished = 0

        for t in range(n_max_loop):
            print(t)
            if t == 0:
                prev = start_phoneme_index

                output = self.decoder(enc_output, prev, mode=mode)  # (B, C, T)
                output = output[..., -1].unsqueeze(-1)    # (B, C, 1)

                candidates, candidates_index = torch.topk(output, beam_size, dim=1)

                for i in range(beam_size):
                    output_list.append([start_phoneme_index, candidates_index[:, i, :]])

            elif t > 0:
                index_list = []
                candidates_dict = {}
                for i in range(beam_size):
                    prev = torch.cat(output_list[i], dim=-1)    # (B, T)
                    if prev[:, -1] == EOS_INDEX:
                        n_finished += 1
                        continue

                    output = self.decoder(enc_output, prev, mode=mode)  # (B, C, T)
                    output = output[..., -1].unsqueeze(-1)      # (B, C, 1)
                    output = torch.softmax(output, dim=1)

                    candidates, candidates_index = torch.topk(output, beam_size, dim=1)
                    index_list.append(i)
                    candidates_dict[i] = (candidates, candidates_index)
                
                final_candidates_list = [[0, 0, 0] for i in range(len(index_list))]

                if len(index_list) != 0:
                    for i in index_list:
                        candidates = candidates_dict[i][0]
                        candidates_index = candidates_dict[i][1]
                        for j in range(candidates.shape[1]):
                            for k in range(len(final_candidates_list)):
                                if final_candidates_list[k][1] < candidates[:, j, :]:
                                    final_candidates_list.insert(k, [i, candidates[:, j, :], candidates_index[:, j, :]])
                                    final_candidates_list.pop(-1)
                                    break

                    for i in range(len(final_candidates_list)):
                        index_prev = final_candidates_list[i][0]
                        index_add = final_candidates_list[i][-1]
                        output_list[index_prev].append(index_add)

                else:
                    break

        final_output_list = []
        for output in output_list:
            output = torch.cat(output[1:], dim=-1)
            final_output_list.append(output)
    
        return final_output_list

    def reset_state(self):
        self.decoder.reset_state()