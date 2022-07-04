import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_subsequent_mask(x):
    len_x = x.shape[-1]
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


def shift(x, n_shift, mode):
    if mode == 'orig':
        pad = (n_shift, 0)
        x = F.pad(x, pad, mode="constant")[:, :, :-n_shift]
    elif mode == "reduction":
        pad = (0, 0, 1, 0)
        x = F.pad(x, pad, mode="constant")[:, :-1, :]
    return x


def posenc(x, device, start_index=0):
    x = x.to('cpu').detach().numpy().copy()
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


class Prenet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=32, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, inner_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(inner_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        音響特徴量をtransformer内の次元に調整する役割

        x : (B, C=feature channels, T)

        return
        y : (B, C=d_model, T)
        """
        x = x.permute(0, -1, -2)  # (B, T, C)
        y = self.fc(x)
        return y.permute(0, -1, -2)   # (B, C, T)


class Postnet(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, n_layers=5, dropout=0.5):
        super().__init__()

        conv = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=5, padding=5//2, bias=False),
            nn.BatchNorm1d(inner_channels),
            nn.Tanh(),
            nn.Dropout(p=dropout)
        )
        for _ in range(n_layers - 2):
            conv.append(nn.Conv1d(
                inner_channels, inner_channels, kernel_size=5, padding=5//2, bias=False
            ))
            conv.append(nn.BatchNorm1d(inner_channels))
            conv.append(nn.Tanh())
            conv.append(nn.Dropout(p=dropout))

        conv.append(nn.Conv1d(inner_channels, out_channels, kernel_size=5, padding=5//2))
        self.conv = conv

    def forward(self, x):
        return x + self.conv(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        print(f"q = {q.shape}")
        print(f"k = {k.shape}")
        print(f"v = {v.shape}")
        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = self.dropout(F.softmax(attention, dim=-1))
        print(f"attention = {attention.shape}")
        output = torch.matmul(attention, v)
        print(f"output = {output.shape}")
        return output, attention


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

    def forward(self, q, k, v, mask=None):
        """
        input : (B, T, C)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   
        q, attn = self.attention(q, k, v, mask=mask)
        # q -> (sz_b, n_head, len_q, d_v)
        # attn -> (sz_b, n_head, len_q, len_k)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))    # (b x lq x (n*dv)) -> (b, lq, d_model)
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, mask=None):
        enc_output, enc_attention = self.attention(enc_input, enc_input, enc_input, mask)
        enc_output = self.fc(enc_output)
        return enc_output, enc_attention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.dec_self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.dec_enc_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_output, self_attention_mask=None, dec_enc_attention_mask=None, mode=None):
        if mode == "inference":
            if self.prev is None:
                self.prev = dec_input
            else:
                self.prev = torch.cat([self.prev, dec_input], dim=1)

            dec_output, dec_self_attention = self.dec_self_attention(dec_input, dec_input, dec_input, self_attention_mask)
            dec_output, dec_enc_attention = self.dec_enc_attention(dec_output, enc_output, enc_output, dec_enc_attention_mask)
            dec_output = self.fc(dec_output[:, -1:, :])
        else:    
            dec_output, dec_self_attention = self.dec_self_attention(dec_input, dec_input, dec_input, self_attention_mask)
            dec_output, dec_enc_attention = self.dec_enc_attention(dec_output, enc_output, enc_output, dec_enc_attention_mask)
            dec_output = self.fc(dec_output)
        return dec_output, dec_self_attention, dec_enc_attention

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
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, lip_feature, data_len=None, max_len=None, return_attention=False):
        if data_len is not None:
            assert max_len is not None
            data_len = torch.div(data_len, self.reduction_factor)
            mask = make_pad_mask(data_len, max_len).to(device=lip_feature.device)
        else:
            mask = None

        lip_feature = self.dropout(lip_feature)
        lip_feature = lip_feature + posenc(lip_feature, device=lip_feature.device, start_index=0)
        lip_feature = lip_feature.permute(0, -1, -2)  # (B, T, C)
        enc_output = self.layer_norm(lip_feature)

        enc_attention_list = []
        for enc_layer in self.enc_layers:
            enc_output, enc_attention = enc_layer(enc_output, mask)
            enc_attention_list.append(enc_attention) if return_attention else []

        if return_attention:
            return enc_output, enc_attention_list
        else:
            return enc_output


class Decoder(nn.Module):
    def __init__(
        self, dec_n_layers, n_head, dec_d_model, pre_in_channels, pre_inner_channels, out_channels,
        n_position, reduction_factor, dropout=0.1, use_gc=False):
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

    def forward(
        self, enc_output, target=None, data_len=None, max_len=None, gc=None, 
        training_method=None, num_passes=None, mixing_prob=None, return_attention=False):
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels

        # target shift
        if training_method is not None:
            target = shift(target, self.reduction_factor, mode="original")

        # reshape for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)  # (B, T, C)
            target = target.reshape(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  # (B, C, T)
        else:
            target = torch.zeros(B, D * self.reduction_factor, 1).to(enc_output.device)
        
        # mask
        if data_len is not None:
            data_len = torch.div(data_len, self.reduction_factor)
            pad_mask = make_pad_mask(data_len, max_len)
            dec_mask = make_pad_mask(data_len, max_len) & get_subsequent_mask(target) # (B, T, T)
        else:
            pad_mask = None
            dec_mask = get_subsequent_mask(target)

        target = self.dropout(self.prenet(target))

        # positional encoding
        if data_len is not None:
            target = target + posenc(target, device=target.device, start_index=0)
            target = self.layer_norm(target.permute(0, -1, -2))
        else:
            if self.start_idx is None:
                self.start_idx = 0
            target = target + posenc(target, device=target.device, start_index=self.start_idx)
            target = self.layer_norm(target.permute(0, -1, -2))
            self.start_idx += 1

        dec_output = target
        dec_self_attention_list = []
        dec_enc_attention_list = []

        # teacher forcing
        if training_method == "tf":
            for dec_layer in self.dec_layers:
                dec_output, dec_self_attention, dec_enc_attention = dec_layer(
                    dec_output, enc_output, self_attention_mask=dec_mask, dec_enc_attention_mask=pad_mask
                )
                dec_self_attention_list.append(dec_self_attention) if return_attention else []
                dec_enc_attention_list.append(dec_enc_attention) if return_attention else []

        # scheduled sampling
        elif training_method == "ss":
            with torch.no_grad():
                for dec_layer in self.dec_layers:
                    dec_output, _, _ = dec_layer(
                        dec_output, enc_output, self_attention_mask=dec_mask, dec_enc_attention_mask=pad_mask
                    )
            # mixing_prob分だけtargetを選択し，それ以外をdec_outputに変更することで混ぜる
            mixing_prob = torch.zeros_like(target) + mixing_prob
            judge = torch.bernoulli(mixing_prob)
            target = torch.where(judge == 1, target, dec_output)
            dec_output = target
            dec_output = shift(dec_output, 1, mode="reduction") 

            for dec_layer in self.dec_layers:
                dec_output, dec_self_attention, dec_enc_attention = dec_layer(
                    dec_output, enc_output, self_attention_mask=dec_mask, dec_enc_attention_mask=pad_mask
                )
                dec_self_attention_list.append(dec_self_attention) if return_attention else []
                dec_enc_attention_list.append(dec_enc_attention) if return_attention else []

        # inference
        else:
            for dec_layer in self.dec_layers:
                dec_output, dec_self_attention, dec_enc_attention = dec_layer(
                    dec_output, enc_output, self_attention_mask=dec_mask, dec_enc_attention_mask=pad_mask
                )
                dec_self_attention_list.append(dec_self_attention) if return_attention else []
                dec_enc_attention_list.append(dec_enc_attention) if return_attention else []

        dec_output = self.conv_o(dec_output.permute(0, -1, -2)) # (B, C, T)
        dec_output = dec_output.permute(0, -1, -2)  # (B, T, C)
        dec_output = dec_output.reshape(B, -1, D)   
        dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)

        if return_attention:
            return dec_output, dec_self_attention_list, dec_enc_attention_list
        else:
            return dec_output
    
    def reset_state(self):
        for dec_layer in self.dec_layers:
            dec_layer.reset_state()
        self.start_idx = None

