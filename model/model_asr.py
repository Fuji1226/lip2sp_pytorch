from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_remake import Encoder, PhonemeDecoder
from data_process.phoneme_encode import SOS_INDEX, EOS_INDEX


class MelEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_conv_layers, dropout):
        super().__init__()
        self.convs = []
        for i in range(n_conv_layers):
            if i == 0:
                in_c = in_channels
            else:
                in_c = hidden_channels
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_c, hidden_channels, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class SpeechRecognizer(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, n_conv_layers, conv_dropout,
        trans_enc_n_layers, trans_enc_n_head, trans_dec_n_layers, trans_dec_n_head,
        out_channels):
        super().__init__()
        self.reduction_factor = int(n_conv_layers * 2)

        self.conv = MelEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_conv_layers=n_conv_layers,
            dropout=conv_dropout,
        )

        self.encoder = Encoder(
            n_layers=trans_enc_n_layers, 
            n_head=trans_enc_n_head, 
            d_model=hidden_channels, 
            reduction_factor=self.reduction_factor,
            pos_max_len=1000,
        )

        self.decoder = PhonemeDecoder(
            dec_n_layers=trans_dec_n_layers,
            n_head=trans_dec_n_head,
            d_model=hidden_channels,
            out_channels=out_channels,
            reduction_factor=self.reduction_factor,
        )

    def forward(self, feature, feature_len, text=None, n_max_loop=1000):
        self.reset_state()

        feature_len = torch.div(feature_len, self.reduction_factor).to(torch.int)
        feature = self.conv(feature)    # (B, C, T)
        enc_output = self.encoder(feature, feature_len)     # (B, T, C)

        if text is not None:
            output = self.decoder_forward(enc_output, feature_len, text)
        else:
            output = self.decoder_inference_greedy(enc_output, feature_len, n_max_loop)
        
        return output
    
    def decoder_forward(self, enc_output, feature_len, text, mode='training'):
        output = self.decoder(enc_output, feature_len, text, mode)
        return output
    
    def decoder_inference_greedy(self, enc_output, feature_len, n_max_loop, mode='inference'):
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

            output = self.decoder(enc_output, feature_len, prev, mode=mode)  # (B, C, T)

            # 最大値のインデックスを取得
            output = output[..., -1].unsqueeze(-1).max(dim=1)[1]   # (B, 1)

            outputs.append(output)


            # もしeosが出たらそこで終了
            if output == EOS_INDEX:
                break
        
        # 最終出力
        output = torch.cat(outputs[1:], dim=-1)     # (B, T)
        return output
    
    def reset_state(self):
        self.decoder.reset_state()