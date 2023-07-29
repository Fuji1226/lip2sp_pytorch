from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.pre_post import Prenet
from data_process.phoneme_encode import IGNORE_INDEX
from utils import check_attention_weight


def get_subsequent_mask(x, diag_mask=False):
    """
    decoderのself attentionを因果的にするためのマスク
    diag_mask = Trueの時,対角成分もTrueになる
    """
    len_x = x.shape[-1]
    if diag_mask:
        subsequent_mask = torch.triu(torch.ones((1, len_x, len_x), device=x.device), diagonal=0).bool()
    else:
        subsequent_mask = torch.triu(torch.ones((1, len_x, len_x), device=x.device), diagonal=1).bool()
    return subsequent_mask.to(device=x.device)


def make_pad_mask(lengths, max_len):
    """
    口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
    マスクする場所をTrue
    """
    # この後の処理でリストになるので先にdeviceを取得しておく
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand     
    return mask.unsqueeze(1).to(device=device)  # (B, 1, T)


def token_mask(x):
    """
    音素ラベルに対してのマスクを作成
    MASK_INDEXに一致するところをマスクする

    x : (B, T)
    mask : (B, T)
    """
    zero_matrix = torch.zeros_like(x)
    one_matrix = torch.ones_like(x)
    mask = torch.where(x == IGNORE_INDEX, one_matrix, zero_matrix).bool() 
    return mask


def shift(x, n_shift):
    """
    因果性を保つためのシフト
    """
    return F.pad(x, (n_shift, 0), mode="constant")[:, :, :-n_shift].clone()


def posenc(x, device, start_index=0):
    """
    x : (B, C, T)
    """
    _, C, T = x.shape

    depth = np.arange(C) // 2 * 2
    depth = np.power(10000.0, depth / C)
    pos = np.arange(start_index, start_index+T)
    phase = pos[:, None] / depth[None]

    phase[:, ::2] += float(np.pi/2)
    positional_encoding = np.sin(phase)

    positional_encoding = positional_encoding.T[None]
    positional_encoding = torch.from_numpy(positional_encoding).to(device)
    positional_encoding = positional_encoding.to(torch.float32)
    return positional_encoding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 300):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x : (T, B, C)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def plot_trans_att_w(att_w, filename, cfg, current_time, ckpt_time):
    for i in range(att_w.shape[0]):
        check_attention_weight(att_w[i], cfg, f"{filename}_head{i}", current_time, ckpt_time)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attention = attention.masked_fill(mask, torch.tensor(float('-inf')))    # maskがTrueを-inf
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask):
        """
        q, k, v : (B, T, C)
        return : (B, T, C)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # (B, T, n_head, C // n_head)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # (B, n_head, T, C // n_head)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        mask = mask.unsqueeze(1)   # 各headに対してブロードキャストするため

        q = self.attention(q, k, v, mask=mask)  # (B, n_head, len_q, d_v)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)    # (b x lq x (n*dv)) -> (b, lq, d_model)
        q = self.dropout(q)
        q += residual
        q = self.layer_norm(q)
        return q


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.w_2(F.relu(self.w_1(x)))
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, mask):
        enc_output = self.attention(enc_input, enc_input, enc_input, mask)
        enc_output = self.fc(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.dec_self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.dec_enc_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_output, self_attention_mask, dec_enc_attention_mask, mode):
        """
        dec_input : (B, T, C)
        enc_output : (B, T, C)
        """
        if mode == "training":
            dec_output = self.dec_self_attention(dec_input, dec_input, dec_input, mask=self_attention_mask)
            dec_output = self.dec_enc_attention(dec_output, enc_output, enc_output, mask=dec_enc_attention_mask)
            dec_output = self.fc(dec_output)

        elif mode == "inference":
            # if self.prev is None:
            #     self.prev = dec_input
            # else:
            #     self.prev = torch.cat([self.prev, dec_input], dim=1)

            # # maskはその都度作成(self.prevのデータ形状を一度変換しなければいけないことに注意)
            # self_attention_mask = get_subsequent_mask(self.prev.permute(0, -1, -2))
            # dec_output = self.dec_self_attention(self.prev, self.prev, self.prev, mask=self_attention_mask)
            # dec_output = self.dec_enc_attention(dec_output, enc_output, enc_output, mask=dec_enc_attention_mask)
            # dec_output = self.fc(dec_output[:, -1:, :])
            
            dec_output = self.dec_self_attention(dec_input, dec_input, dec_input, mask=self_attention_mask)
            dec_output = self.dec_enc_attention(dec_output, enc_output, enc_output, mask=dec_enc_attention_mask)
            dec_output = self.fc(dec_output)
        return dec_output

    def reset_state(self):
        self.prev = None


class Encoder(nn.Module):
    def __init__(self, n_layers, n_head, d_model, reduction_factor, pos_max_len, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4
        self.reduction_factor = reduction_factor

        self.pos_encoder = PositionalEncoding(d_model, max_len=pos_max_len)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        """
        mask = make_pad_mask(data_len, x.shape[-1])

        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.layer_norm(x).permute(1, 0, 2)     # (T, B, C)
        x = self.pos_encoder(x).permute(1, 0, 2)    # (B, T, C)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self, dec_n_layers, n_head, dec_d_model, pre_in_channels, pre_inner_channels, out_channels,
        n_position, reduction_factor, dropout=0.1):
        super().__init__()
        self.d_k = dec_d_model // n_head
        self.d_v = dec_d_model // n_head
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.d_inner = dec_d_model * 4

        self.prenet = Prenet(pre_in_channels, dec_d_model, pre_inner_channels)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(dec_d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(dec_n_layers)
        ])
        self.conv_o = nn.Conv1d(dec_d_model, self.out_channels * self.reduction_factor, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dec_d_model, eps=1e-6)

    def forward(self, enc_output, target=None, data_len=None, gc=None, mode=None):
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels

        # target shift
        if mode == "training":
            target = shift(target, self.reduction_factor)

        # view for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)  # (B, T, C)
            target = target.contiguous().view(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  # (B, C, T)
        else:
            target = torch.zeros(B, D * self.reduction_factor, 1).to(device=enc_output.device, dtype=enc_output.dtype) 
        
        # mask
        if mode == "training":
            data_len = torch.div(data_len, self.reduction_factor)
            max_len = T
            dec_enc_attention_mask = make_pad_mask(data_len, max_len).repeat(1, max_len, 1)
            self_attention_mask = make_pad_mask(data_len, max_len).repeat(1, max_len, 1) | get_subsequent_mask(target).repeat(B, 1, 1) # (B, T, T)

        elif mode == "inference":
            dec_enc_attention_mask = None
            self_attention_mask = None

        # prenet
        target = self.dropout(self.prenet(target))

        # positional encoding & decoder layers
        if mode == "training":
            target = target + posenc(target, device=target.device, start_index=0)
            target = self.layer_norm(target.permute(0, -1, -2))     # (B, T, C)
            dec_layer_out = target
            for dec_layer in self.dec_layers:
                dec_layer_out = dec_layer(dec_layer_out, enc_output, self_attention_mask, dec_enc_attention_mask, mode)

        elif mode == "inference":
            if self.start_idx is None:
                self.start_idx = 0
            target = target + posenc(target, device=target.device, start_index=self.start_idx)
            target = self.layer_norm(target.permute(0, -1, -2))     # (B, T, C)
            self.start_idx += 1
            dec_layer_out = target
            for dec_layer in self.dec_layers:
                dec_layer_out = dec_layer(dec_layer_out, enc_output, self_attention_mask, dec_enc_attention_mask, mode)

        dec_output = self.conv_o(dec_layer_out.permute(0, -1, -2))      # (B, C, T)
        dec_output = dec_output.permute(0, -1, -2)  # (B, T, C)
        dec_output = dec_output.contiguous().view(B, -1, D)
        dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)
        return dec_output
    
    def reset_state(self):
        for dec_layer in self.dec_layers:
            dec_layer.reset_state()
        self.start_idx = None


class PhonemeDecoder(nn.Module):
    def __init__(self, dec_n_layers, n_head, d_model, out_channels, reduction_factor, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.d_inner = d_model * 4

        # token embedding
        self.emb_layer = nn.Embedding(out_channels, d_model, padding_idx=IGNORE_INDEX)

        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(dec_n_layers)
        ])

        self.out_fc = nn.Linear(d_model, out_channels)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_output, data_len, prev, mode):
        """
        enc_output : (B, T, C)
        prev(phoneme sequence) : (B, T)

        return
        output : (B, C, T)
        """
        data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
        max_len = enc_output.shape[1]

        # パディングした部分に対してのマスク
        dec_enc_attention_mask = make_pad_mask(data_len, max_len).to(device=enc_output.device)    # (B, 1, len_enc)

        # self attentionを因果的にするため + パディングした部分に対してのマスク
        self_attention_mask = token_mask(prev).unsqueeze(1) | get_subsequent_mask(prev)  # (B, len_prev, len_prev)

        prev_emb = self.emb_layer(prev)     # (B, T, C)

        # positional encoding & decoder layer
        prev_emb = prev_emb.permute(0, 2, 1)    # (B, C, T)
        prev_emb = prev_emb + posenc(prev_emb, device=prev_emb.device, start_index=0)
        prev_emb = self.layer_norm(prev_emb.permute(0, 2, 1))    # (B, T, C)
        dec_layer_out = prev_emb

        for dec_layer in self.dec_layers:
            dec_layer_out = dec_layer(dec_layer_out, enc_output, self_attention_mask, dec_enc_attention_mask, mode)

        output = self.out_fc(dec_layer_out)
        output = output.permute(0, 2, 1)    # (B, T, C)
        assert output.shape[-1] == prev.shape[-1]
        return output

    def reset_state(self):
        for dec_layer in self.dec_layers:
            dec_layer.reset_state()
        self.start_idx = None


class OfficialEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * 4),
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
    def forward(self, x, data_len=None):
        B, C, T = x.shape
        x = x.permute(2, 0, 1)
        if data_len is not None:
            data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
            pad_mask = make_pad_mask(data_len, T).squeeze(1)
        else:
            pad_mask = None
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        out = out.permute(1, 0, 2)
        return out