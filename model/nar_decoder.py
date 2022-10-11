import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.grad_reversal import GradientReversal
from model.transformer_remake import Encoder
from utils import count_params


class TCDecoder(nn.Module):
    def __init__(self, cond_channels, out_channels, inner_channels, n_layers, kernel_size, dropout):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose1d(cond_channels, inner_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
        )

        padding = (kernel_size - 1) // 2
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm1d(inner_channels),
                nn.ReLU(),
            ) for _ in range(n_layers)
        ])

        self.out_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_output):
        """
        enc_output : (B, T, C)
        """
        enc_output = enc_output.permute(0, -1, 1)   # (B, C, T)
        output = enc_output

        # 音響特徴量までアップサンプリング
        output = self.upsample_layer(output)
        
        for layer in self.conv_layers:
            output = self.dropout(output)
            output = layer(output)
        
        output = self.out_layer(output)
        return output


class GatedConvLayer(nn.Module):
    def __init__(self, cond_channels, inner_channels, kernel_size, dropout=0.5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv1d(inner_channels, inner_channels * 2, kernel_size=kernel_size, padding=padding)
        self.cond_layer = nn.Conv1d(cond_channels, inner_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        """
        x : (B, C, T)
        enc_output : (B, C, T)
        """
        res = x

        y = self.dropout(x)
        y = self.conv_layer(y)      # (B, 2 * C, T)
        y1, y2 = torch.split(y, y.shape[1] // 2, dim=1)

        # repeatでフレームが連続するように、あえてrepeat_interleaveを使用しています
        enc_output = enc_output.repeat_interleave(2, dim=-1)     # (B, C, 2 * T)
        enc_output = F.softsign(self.cond_layer(enc_output))
        y2 = y2 + enc_output

        out = torch.sigmoid(y1) * y2
        out += res
        return out


class GatedTCDecoder(nn.Module):
    """
    ゲート付き活性化関数を利用したdecoder
    """
    def __init__(self, cond_channels, out_channels, inner_channels, n_layers, kernel_size, dropout=0.5):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose1d(cond_channels, inner_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
        )

        self.conv_layers = nn.ModuleList([
            GatedConvLayer(cond_channels, inner_channels, kernel_size, dropout) for _ in range(n_layers)
        ])

        self.out_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

    def forward(self, enc_output):
        """
        enc_output : (B, T, C)
        """
        enc_output = enc_output.permute(0, -1, 1)   # (B, C, T)
        output = enc_output

        # 音響特徴量までアップサンプリング
        output = self.upsample_layer(output)

        for layer in self.conv_layers:
            output = layer(output, enc_output)
        
        output = self.out_layer(output)
        return output


class ResBlock(nn.Module):
    def __init__(self, inner_channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_layers = nn.Sequential(
            nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = x
        out = self.conv_layers(x)
        return out + res


class FeadAddPredicter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.out_layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        x : (B, C, T)
        out : (B, C, T)
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.out_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class PhonemePredicter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        x : (B, C, T)
        out : (B, C, T)
        """
        out = x.permute(0, 2, 1)
        out = self.layer(out)
        return out.permute(0, 2, 1)


class ResTCDecoder(nn.Module):
    """
    残差結合を取り入れたdecoder
    """
    def __init__(
        self, cond_channels, out_channels, inner_channels, n_layers, kernel_size, dropout, 
        feat_add_channels, feat_add_layers, use_feat_add, phoneme_classes, use_phoneme, spk_emb_dim, 
        n_attn_layer, n_head, d_model, reduction_factor,  use_attention,
        upsample_method, compress_rate):
        super().__init__()
        self.use_phoneme = use_phoneme
        self.upsample_method = upsample_method
        self.compress_rate = compress_rate
        self.use_attention = use_attention

        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose1d(cond_channels, inner_channels, kernel_size=compress_rate, stride=compress_rate),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.interp_layer = nn.Conv1d(cond_channels, inner_channels, kernel_size=1)

        self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        self.feat_add_layer = FeadAddPredicter(inner_channels, feat_add_channels, kernel_size=3, n_layers=feat_add_layers, dropout=dropout)
        self.adjust_feat_add_layer = nn.Conv1d(feat_add_channels, inner_channels, kernel_size=1)
        self.phoneme_layer = PhonemePredicter(inner_channels, phoneme_classes)
        self.gr_layer = GradientReversal(1.0)

        if self.use_attention:
            self.pre_attention_layer = nn.Conv1d(inner_channels, d_model, kernel_size=1)
            self.attention = Encoder(
                n_layers=n_attn_layer,
                n_head=n_head,
                d_model=d_model,
                reduction_factor=reduction_factor,
            )
            self.post_attention_layer = nn.Conv1d(d_model, inner_channels, kernel_size=1)

        self.conv_layers = nn.ModuleList(
            ResBlock(inner_channels, kernel_size, dropout) for _ in range(n_layers)
        )

        self.out_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

    def forward(self, enc_output, spk_emb=None, data_len=None):
        """
        enc_outout : (B, T, C)
        spk_emb : (B, C)
        """
        feat_add_out = phoneme = None

        enc_output = enc_output.permute(0, -1, 1)   # (B, C, T)
        out = enc_output

        # 音響特徴量のフレームまでアップサンプリング
        if self.upsample_method == "conv":
            out_upsample = self.upsample_layer(out)
        elif self.upsample_method == "interpolate":
            out_upsample = F.interpolate(out ,scale_factor=self.compress_rate)
            out_upsample = self.interp_layer(out_upsample)

        out = out_upsample

        # speaker embedding
        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(-1).expand(-1, -1, out.shape[-1])   # (B, C) -> (B, C, T)
            out = torch.cat([out, spk_emb], dim=1)
            out = self.spk_emb_layer(out)

        for layer in self.conv_layers:
            out = layer(out)

        out = self.out_layer(out)
        return out, feat_add_out, phoneme, out_upsample


if __name__ == "__main__":
    inner_channels = 128
    kernel_size = 3
    n_layers = 2

    