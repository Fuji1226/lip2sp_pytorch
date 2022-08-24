"""
transformerのattentionを田口さんと一緒にしてみたやつです
特に変化がなかったし,conformerと出力のデータ形状が違ってややこしいので使わなくなりました
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_subsequent_mask(x, diag_mask):
    len_x = x.shape[-1]
    if diag_mask:
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_x, len_x), device=x.device), diagonal=0)).bool()
    else:
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_x, len_x), device=x.device), diagonal=1)).bool()
    return subsequent_mask.to(device=x.device)


def make_pad_mask(lengths, max_len):
    device = lengths.device
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand     
    return mask.unsqueeze(1).to(device=device)


def shift(x, n_shift):
    return F.pad(x, (n_shift, 0), mode="constant")[:, :, :-n_shift].clone()


def posenc(x, device, start_index=0):
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


class LayerNorm1D(nn.LayerNorm):
    def __init__(self, n_dim, eps=1e-6):
        super().__init__(normalized_shape=n_dim, eps=eps)
    
    def forward(self, x):
        B, C, T = x.shape
        x = x.permute(0, -1, -2)    # (B, T, C)
        # x = x.view(B * T, C)
        out = super().forward(x)
        # x = x.view(B, T, C)
        out = out.permute(0, -1, -2)    # (B, C, T)
        return out


class Prenet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=32, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(in_channels, inner_channels),
            nn.Conv1d(in_channels, inner_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(inner_channels, out_channels),
            nn.Conv1d(inner_channels, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        音響特徴量をtransformer内の次元に調整する役割

        x : (B, C=feature channels, T)

        return
        y : (B, C=d_model, T)
        """
        y = self.fc(x)
        return y


class Postnet(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, post_n_layers=5, dropout=0.5):
        super().__init__()

        conv = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm1d(inner_channels),
            nn.Tanh(),
            nn.Dropout(p=dropout)
        )
        for _ in range(post_n_layers - 2):
            conv.append(nn.Conv1d(
                inner_channels, inner_channels, kernel_size=5, padding=2, bias=True
            ))
            conv.append(nn.BatchNorm1d(inner_channels))
            conv.append(nn.Tanh())
            conv.append(nn.Dropout(p=dropout))

        conv.append(nn.Conv1d(inner_channels, out_channels, kernel_size=5, padding=2, bias=True))
        self.conv = conv

    def forward(self, x):
        return x + self.conv(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head

        self.W_QKV = nn.Conv1d(d_model, d_model * 3, kernel_size=1, bias=False)
        self.W_Q = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.W_KV = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=False)
        self.fc = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm = LayerNorm1D(d_model)

    def forward(self, x, y=None, mask=None):
        """
        input : (B, C, T)
        return : (B, C, T)
        """
        residual = x
        if y is None:
            qkv = self.W_QKV(x)
            (Q, K, V) = torch.split(qkv, qkv.shape[1] // 3, dim=1)
        else:
            Q = self.W_Q(x)
            kv = self.W_KV(y)
            (K, V) = torch.split(kv, kv.shape[1] // 2, dim=1)
        batch, d_model, n_query = Q.shape
        _, _, n_key = K.shape

        # print("----- Q -----")
        # print(f"Q.grad = {Q.grad}")
        # print(f"Q.requires_grad = {Q.requires_grad}")
        # print(f"Q.is_leaf = {Q.is_leaf}")

        batch_Q = torch.cat(torch.split(Q, Q.shape[1] // self.n_head, dim=1), dim=0)
        batch_K = torch.cat(torch.split(K, K.shape[1] // self.n_head, dim=1), dim=0)
        batch_V = torch.cat(torch.split(V, V.shape[1] // self.n_head, dim=1), dim=0)
        assert batch_Q.shape == (batch * self.n_head, d_model // self.n_head, n_query)
        assert batch_K.shape == (batch * self.n_head, d_model // self.n_head, n_key)
        assert batch_V.shape == (batch * self.n_head, d_model // self.n_head, n_key)

        # print("----- batch_Q -----")
        # print(f"batch_Q.grad = {batch_Q.grad}")
        # print(f"batch_Q.requires_grad = {batch_Q.requires_grad}")
        # print(f"batch_Q.is_leaf = {batch_Q.is_leaf}")
        batch_A = torch.matmul(batch_Q.transpose(1, 2), batch_K)
        
        if mask is not None:
            # print(f"mask = {mask.shape}")
            mask = torch.cat([mask] * self.n_head, dim=0)
            # print(f"mask = {mask.shape}")
            batch_A = batch_A.masked_fill(mask == 0, torch.tensor(float('-inf')))

        # print("----- batch_A after mask -----")
        # print(f"batch_A.grad = {batch_A.grad}")
        # print(f"batch_A.requires_grad = {batch_A.requires_grad}")
        # print(f"batch_A.is_leaf = {batch_A.is_leaf}")
    
        batch_A = torch.softmax(batch_A, dim=2)
        batch_A = self.dropout(batch_A)
        assert batch_A.shape == (batch * self.n_head, n_query, n_key)

        batch_A = batch_A.unsqueeze(1).expand(-1, batch_V.shape[1], -1, -1)
        batch_V = batch_V.unsqueeze(2).expand(-1, -1, batch_A.shape[2], -1)
        # print("----- batch_A after expand -----")
        # print(f"batch_A.grad = {batch_A.grad}")
        # print(f"batch_A.requires_grad = {batch_A.requires_grad}")
        # print(f"batch_A.is_leaf = {batch_A.is_leaf}")
        batch_C = torch.sum(batch_A * batch_V, dim=3)
        C = torch.cat(torch.split(batch_C, batch_C.shape[0] // self.n_head, dim=0), dim=1)
        assert C.shape == Q.shape

        # print("----- C after expand -----")
        # print(f"C.grad = {C.grad}")
        # print(f"C.requires_grad = {C.requires_grad}")
        # print(f"C.is_leaf = {C.is_leaf}")
        C = self.dropout(self.fc(C))
        C += residual
        C = self.layer_norm(C)
        return C


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=1)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=1)
        # self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.layer_norm = LayerNorm1D(d_in)
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

    def forward(self, enc_input, mask=None):
        attention_out = self.attention(x=enc_input, mask=mask)
        enc_output = self.fc(attention_out)
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, diag_mask=False):
        super().__init__()
        self.diag_mask = diag_mask
        self.dec_self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.dec_enc_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_output, self_attention_mask=None, dec_enc_attention_mask=None, mode=None):
        if mode == "training":
            out_self_atten = self.dec_self_attention(x=dec_input, mask=self_attention_mask)
            # print(f"enc_output = {enc_output}")
            # print(f"dec_output = {dec_output}")
            # print(f"out_self_atten_mean = {torch.mean(out_self_atten ** 2)}")
            # print(f"enc_outout_mean = {torch.mean(enc_output ** 2)}")
            # print(f"out_self_atten = {out_self_atten}")
            # print(f"enc_output = {enc_output}")
            out_dec_enc_atten = self.dec_enc_attention(x=out_self_atten, y=enc_output, mask=dec_enc_attention_mask)
            dec_output = self.fc(out_dec_enc_atten)

        elif mode == "inference":
            if self.prev is None:
                self.prev = dec_input
            else:
                self.prev = torch.cat([self.prev, dec_input], dim=-1)
            #############################################
            self_attention_mask = get_subsequent_mask(self.prev, self.diag_mask)
            #############################################
            out_self_atten = self.dec_self_attention(x=self.prev, mask=self_attention_mask)
            # print(f"enc_output = {enc_output}")
            # print(f"dec_output = {dec_output}")
            # print(f"out_self_atten_mean = {torch.mean(out_self_atten ** 2)}")
            # print(f"enc_outout_mean = {torch.mean(enc_output ** 2)}")
            # print(f"out_self_atten = {out_self_atten}")
            # print(f"enc_output = {enc_output}")
            out_dec_enc_atten = self.dec_enc_attention(x=out_self_atten, y=enc_output, mask=dec_enc_attention_mask)
            # print(f"dec_output = {dec_output.shape}")
            # print(f"dec_output = {dec_output}")
            # print(f"dec_output[:, :, -1:] = {dec_output[:, :, -1:].shape}")
            # print(f"dec_output[:, :, -1:] = {dec_output[:, :, -1:]}")
            # print(f"dec_output[:, :, -1] = {dec_output[:, :, -1].shape}")
            # print(f"dec_output[:, :, -1] = {dec_output[:, :, -1]}")
            dec_output = self.fc(out_dec_enc_atten[:, :, -1:])
            # dec_output = self.fc(dec_output)
            # print(f"dec_output = {dec_output.shape}")
            # print(f"dec_output = {dec_output}")

            # self_attention_mask = get_subsequent_mask(dec_input, self.diag_mask)
            # out_self_atten = self.dec_self_attention(x=self.prev, mask=self_attention_mask)
            # out_dec_enc_atten = self.dec_enc_attention(x=out_self_atten, y=enc_output, mask=dec_enc_attention_mask)
            # dec_output = self.fc(out_dec_enc_atten)
        return dec_output

    def reset_state(self):
        self.prev = None


class Encoder(nn.Module):
    def __init__(self, n_layers, n_head, d_model, n_position, reduction_factor, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4
        self.reduction_factor = reduction_factor

        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm = LayerNorm1D(d_model)

    def forward(self, lip_feature, data_len=None, max_len=None):
        # print("----- encoder start -----")
        if data_len is not None:
            assert max_len is not None
            data_len = torch.div(data_len, self.reduction_factor)
            mask = make_pad_mask(data_len, max_len).to(device=lip_feature.device)
            ###################################
            # mask = None
            ###################################
            # print(f"mask = {mask.shape}")
            # print(f"mask = {mask}")
        else:
            mask = None
        # print(f"lip_feature = {lip_feature.shape}")
        lip_feature = self.dropout(lip_feature)
        lip_feature = lip_feature + posenc(lip_feature, device=lip_feature.device, start_index=0)
        enc_output = self.layer_norm(lip_feature)

        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, mask)    
        return enc_output


class Decoder(nn.Module):
    def __init__(
        self, dec_n_layers, n_head, dec_d_model, pre_in_channels, pre_inner_channels, out_channels,
        n_position, reduction_factor, dropout=0.1, use_gc=False, diag_mask=False):
        super().__init__()
        self.d_k = dec_d_model // n_head
        self.d_v = dec_d_model // n_head
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.d_inner = dec_d_model * 4
        self.diag_mask = diag_mask

        self.prenet = Prenet(pre_in_channels, dec_d_model, pre_inner_channels)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(dec_d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout, diag_mask=self.diag_mask)
            for _ in range(dec_n_layers)
        ])
        self.conv_o = nn.Conv1d(dec_d_model, self.out_channels * self.reduction_factor, kernel_size=1)
        # self.layer_norm = nn.LayerNorm(dec_d_model, eps=1e-6)
        self.layer_norm = LayerNorm1D(dec_d_model)

    def forward(self, enc_output, target=None, data_len=None, max_len=None, gc=None, mode=None):
        # print("\n----- decoder start -----")
        assert mode == "training" or "inference"
        B = enc_output.shape[0]
        T = enc_output.shape[-1]
        D = self.out_channels

        # print("----- target input -----")
        # print(f"target.grad = {target.grad}")
        # print(f"target.requires_grad = {target.requires_grad}")
        # print(f"target.is_leaf = {target.is_leaf}")

        # target shift
        if mode == "training":
            target = shift(target, self.reduction_factor)

        # print("----- target after shift -----")
        # print(f"target.grad = {target.grad}")
        # print(f"target.requires_grad = {target.requires_grad}")
        # print(f"target.is_leaf = {target.is_leaf}")

        # view for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)  # (B, T, C)
            target = target.contiguous().view(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  # (B, C, T)
        else:
            target = torch.zeros(B, D * self.reduction_factor, 1).to(device=enc_output.device, dtype=enc_output.dtype) 
        
        # mask
        if mode == "training":
            assert data_len is not None and max_len is not None
            data_len = torch.div(data_len, self.reduction_factor)
            ############################
            pad_mask = make_pad_mask(data_len, max_len)
            dec_mask = make_pad_mask(data_len, max_len) & get_subsequent_mask(target, self.diag_mask) # (B, T, T)
            # pad_mask = None
            # dec_mask = get_subsequent_mask(target, self.diag_mask)
            ############################
            # print(f"pad_mask = {pad_mask.shape}")
            # print(f"pad_mask = {pad_mask}")
            # print(f"dec_mask = {dec_mask.shape}")
            # print(f"dec_mask = {dec_mask}")
        elif mode == "inference":
            assert data_len is None and max_len is None
            pad_mask = None
            dec_mask = None

        # prenet
        target = self.dropout(self.prenet(target))

        # print("----- target after prenet -----")
        # print(f"target.grad = {target.grad}")
        # print(f"target.requires_grad = {target.requires_grad}")
        # print(f"target.is_leaf = {target.is_leaf}")

        # positional encoding & decoder layers
        if mode == "training":
            target = target + posenc(target, device=target.device, start_index=0)
            target = self.layer_norm(target)
            dec_layer_out = target
            for dec_layer in self.dec_layers:
                dec_layer_out = dec_layer(dec_layer_out, enc_output, self_attention_mask=dec_mask, dec_enc_attention_mask=pad_mask, mode=mode)

        elif mode == "inference":
            if self.start_idx is None:
                self.start_idx = 0
            target = target + posenc(target, device=target.device, start_index=self.start_idx)
            target = self.layer_norm(target)
            self.start_idx += 1
            dec_layer_out = target
            for dec_layer in self.dec_layers:
                dec_layer_out = dec_layer(dec_layer_out, enc_output, self_attention_mask=dec_mask, dec_enc_attention_mask=pad_mask, mode=mode)

        # print("----- dec_output after dec_layers -----")
        # print(f"dec_output.grad = {dec_output.grad}")
        # print(f"dec_output.requires_grad = {dec_output.requires_grad}")
        # print(f"dec_output.is_leaf = {dec_output.is_leaf}")
        dec_output = self.conv_o(dec_layer_out)

        # print("----- dec_output after conv_o -----")
        # print(f"dec_output.grad = {dec_output.grad}")
        # print(f"dec_output.requires_grad = {dec_output.requires_grad}")
        # print(f"dec_output.is_leaf = {dec_output.is_leaf}")
        dec_output = dec_output.permute(0, -1, -2)  # (B, T, C)
        dec_output = dec_output.contiguous().view(B, -1, D)   
        dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)
        return dec_output
    
    def reset_state(self):
        for dec_layer in self.dec_layers:
            dec_layer.reset_state()
        self.start_idx = None
