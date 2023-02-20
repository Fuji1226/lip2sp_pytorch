from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import ResNet3D
from model.transformer_remake import Encoder, PhonemeDecoder
from data_process.phoneme_encode import SOS_INDEX, EOS_INDEX


class Lip2Text(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_inner_channels,
        trans_enc_n_layers, trans_enc_n_head, trans_dec_n_layers, trans_dec_n_head,
        res_dropout, reduction_factor, which_encoder, use_ctc):
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

        if use_ctc:
            self.ctc_output_layer = nn.Linear(inner_channels, out_channels)

        # decoder
        self.decoder = PhonemeDecoder(
            dec_n_layers=trans_dec_n_layers,
            n_head=trans_dec_n_head,
            d_model=inner_channels,
            out_channels=out_channels,
            reduction_factor=reduction_factor,
        )

    def forward(self, lip, data_len, prev=None, training_method=None, mixing_prob=None, n_max_loop=None, search_method=None, beam_size=5):
        # 推論時にdecoderでインスタンスとして保持されていた結果の初期化
        self.reset_state()

        # encoder
        enc_output, fmaps = self.ResNet_GAP(lip)
        
        enc_output = self.encoder(enc_output, data_len)    # (B, T, C)

        if hasattr(self, "ctc_output_layer"):
            ctc_output = self.ctc_output_layer(enc_output)  # (B, T, C)
        else:
            ctc_output = None

        # decoder
        # train
        if prev is not None:
            if training_method == "tf":
                output = self.decoder_forward(enc_output, data_len, prev)

            elif training_method == "ss":
                with torch.no_grad():
                    output = self.decoder_forward(enc_output, data_len, prev)

                    # 最大値(Onehotの1のところ)のインデックスを取得
                    output = output.max(dim=1)[1]   # (B, T)

                    # mixing_prob分だけラベルを選択し，それ以外を変更することで混ぜる
                    mixing_prob = torch.zeros_like(prev) + mixing_prob
                    judge = torch.bernoulli(mixing_prob)
                    mixed_prev = torch.where(judge == 1, prev, output)

                # 混ぜたやつでもう一回計算してそれを出力とする
                output = self.decoder_forward(enc_output, data_len, mixed_prev)
        # inference
        else:
            if search_method == "greedy":
                output = self.decoder_inference_greedy(enc_output, data_len, n_max_loop)
            elif search_method == "beam_search":
                output = self.decoder_inference_beam_search(enc_output, data_len, n_max_loop, beam_size=beam_size)

        return output, ctc_output

    def decoder_forward(self, enc_output, data_len, prev, mode="training"):
        """
        学習時の処理
        """
        output = self.decoder(enc_output, data_len, prev, mode=mode)    # (B, C, T)
        return output

    def decoder_inference_greedy(self, enc_output, data_len, n_max_loop, mode="inference"):
        """
        greedy searchによる推論

        enc_output : (B, T, C)
        n_max_loop : eosが出なくてloopが終わらなくなる可能性があるので,上限を定めるための値
        """
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

            output = self.decoder(enc_output, data_len, prev, mode=mode)  # (B, C, T)

            # 最大値のインデックスを取得
            output = output[..., -1].unsqueeze(-1).max(dim=1)[1]   # (B, 1)

            outputs.append(output)

            # もしeosが出たらそこで終了
            if output == EOS_INDEX:
                break
        
        # 最終出力
        output = torch.cat(outputs[1:], dim=-1)     # (B, T)
        return output

    def decoder_inference_beam_search(self, enc_output, data_len, n_max_loop, beam_size, mode="inference"):
        """
        beam searchによる推論

        enc_output : (B, T, C)
        n_max_loop : eosが出なくてloopが終わらなくなる可能性があるので,上限を定めるための値
        """
        B, T, C = enc_output.shape 

        start_phoneme_index = torch.zeros(B, 1).to(device=enc_output.device, dtype=torch.long)
        start_phoneme_index[:] = SOS_INDEX   

        output_list = []
        log_probs = torch.zeros(beam_size)

        for t in range(n_max_loop):
            if t == 0:
                prev = start_phoneme_index

                output = self.decoder(enc_output, data_len, prev, mode=mode)  # (B, C, T)
                output = output[0, :, -1]   # (C,)
                output = torch.log_softmax(output, dim=0)

                # topkで上位beam_size個の候補を選択
                candidates, candidates_index = torch.topk(output, beam_size)    # (beam_size,), (beam_size,)

                # 対数確率を保持
                log_probs = candidates

                for i in range(beam_size):
                    output_list.append([torch.tensor(SOS_INDEX), candidates_index[i]])

                # beam_size分をバッチとして並列処理するためにexpandしておく
                enc_output = enc_output.expand(beam_size, -1, -1)

            if t > 0:
                prev_list = []
                for i in range(beam_size):
                    prev = torch.stack(output_list[i])  # (T,)
                    prev_list.append(prev)
                prev = torch.stack(prev_list, dim=0)    # (B, T)

                output = self.decoder(enc_output, data_len, prev, mode=mode)  # (B, C, T)
                output = output[..., -1]    # (B, C)
                output = torch.log_softmax(output, dim=1)

                # 以前の対数確率も考慮する
                output = output + log_probs.unsqueeze(1)

                # バッチ全体からtopkでbeam_size分候補を確保
                output = output.reshape(-1)     # (B * C,)
                candidates, candidates_index = torch.topk(output, beam_size)

                # すでにEOSを出しているならEOSに変更
                for i in range(beam_size):
                    prev_last = prev[i, -1]
                    if prev_last == EOS_INDEX:
                        candidates_index[i] = EOS_INDEX
                
                # 対数確率を更新
                log_probs = candidates

                # 前時刻の出力のどれに対応するかを表すインデックス
                prev_indices = candidates_index // self.out_channels

                # token(音素の種類)を表すインデックス
                token_indices = candidates_index % self.out_channels

                n_finished = 0
                output_list_updated = []
                for prev_index, token_index in zip(prev_indices, token_indices):
                    # prev_indexに対応した前時刻の出力のリスト
                    out_list = []
                    for output in output_list[prev_index]:
                        out_list.append(output)

                    # token_indexがEOSになった候補の個数をカウント
                    if token_index == EOS_INDEX:
                        n_finished += 1

                    out_list.append(token_index)
                    output_list_updated.append(out_list)

                # 出力リスト更新
                output_list = output_list_updated

                # 全ての候補がEOSになっていれば抜ける
                if n_finished == beam_size:
                    break
        
        final_output_list = []
        for i in range(beam_size):
            final_output_list.append(
                torch.stack(output_list[i][1:])     # sosは除く。(T,)
            )

        return final_output_list


    def reset_state(self):
        self.decoder.reset_state()