"""
reference
https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
"""
import os
import sys

from zmq import device
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils import weight_norm
import numpy as np


def get_subsequent_mask(seq):
    ''' 
    音響特徴量の未来の値のマスク
    推論時と条件を合わせるためのdecoderに対する処置

    For masking out the subsequent info. 
    '''
    len_s = seq.size()[-1]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def make_pad_mask(lengths, max_len):
    """
    data_lenが短くて0パディングしたデータに対してのマスク
    maxlenは口唇動画のフレーム数
    
    Make mask for padding frames

    Args:
        lengths (list): list of lengths（tensorでOK）
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand     # (B, T=maxlen)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return mask.unsqueeze(1).to(device)    # (B, 1, T)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # tempratureは√d_k
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)    # maskがFalseのところを-infに変更
        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

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
        input
        (B, T, C)

        return

        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # q -> (sz_b, n_head, len_q, d_v)
        # attn -> (sz_b, n_head, len_q, len_k)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))    # (b x lq x (n*dv)) -> (b, lq, d_model)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):

        residual = x

        # x = self.w_2(F.relu(self.w_1(x)))
        x = self.w_2(self.activation(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        # enc_output : (b, lq, d_model)
        # ouc_slf_attn : (sz_b, n_head, len_q, len_k)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output : (b lq, d_model) -> (b lq, d_inner) -> (b lq, d_model)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        
        # 2層目のattentionでは、decoder1層目からの出力をquery、encoderからの出力をkey、valueとする。
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)   # (1, n_position, d_hid)

    def forward(self, x):
        """
        x : (B, T, d_hid)
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()   


class Encoder(nn.Module):
    def __init__(
        self,n_layers, n_head, d_model, 
        n_position, reduction_factor, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4

        self.position_enc = PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model
        self.reduction_factor = reduction_factor
    
    def forward(self, prenet_out, data_len=None, max_len=None, return_attns=False):
        """
        prenet_out : (B, C, T)
        """
        # get mask（学習時のみ、0パディングされた部分を隠すためのマスクを作成）
        if data_len is not None:
            assert max_len is not None
            data_len = torch.div(data_len, self.reduction_factor)
            mask = make_pad_mask(data_len, max_len)
        else:
            # 推論時はマスクなし
            mask = None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.to(device)

        # positional encoding
        prenet_out = prenet_out.permute(0, -1, -2)  # (B, T, C)
        prenet_out = self.dropout(self.position_enc(prenet_out))
        enc_outout = self.layer_norm(prenet_out)

        enc_slf_attn_list = []
        # encoder layers
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_outout, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Prenet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=32, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            weight_norm(nn.Linear(in_channels, inner_channels)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            weight_norm(nn.Linear(inner_channels, out_channels)),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (B, C=feature channels, T)

        return
        y : (B, C=d_model, T)
        """
        x = x.permute(0, -1, -2)  # (B, T, C)
        y = self.fc(x)
        return y.permute(0, -1, -2)   # (B, C, T)


class Decoder(nn.Module):
    def __init__(
        self, n_layers, n_head, d_model, 
        pre_in_channels, pre_inner_channels, out_channels, 
        n_position, reduction_factor, dropout=0.1, use_gc=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.d_inner = d_model * 4

        self.prenet = Prenet(pre_in_channels, d_model, pre_inner_channels)

        self.position_enc = PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])

        self.conv_o = weight_norm(nn.Conv1d(d_model, self.out_channels * self.reduction_factor, kernel_size=1))

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

        # self.proj_pre = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        # if use_gc:
        #     self.proj_gc = nn.Linear()

    def forward(
        self, enc_output, data_len, max_len, target=None, gc=None, return_attns=False, 
        training_method=None, num_passes=None, mixing_prob=None):
        """
        reduction_factorにより、
        口唇動画のフレームの、reduction_factor倍のフレームを同時に出力する

        input
        target（音響特徴量） : (B, C, T)
        enc_output : (B, T, C)

        return
        dec_output : (B, C, T)
        """
        assert training_method == 'tf' or training_method == "ss", "please set traning_method"
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels
        
        # global conditionの結合
        
        # reshape for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)
            target = target.reshape(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  
        
        # get target_mask
        data_len = torch.div(data_len, self.reduction_factor)
        mask = make_pad_mask(data_len, max_len)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.to(device)
        target = target.to(device)
        data_len = data_len.to(device)
        
        max_len = torch.tensor(max_len).to(device)
        target_mask = make_pad_mask(data_len, max_len) & get_subsequent_mask(target) # (B, T, T)
        
        # Prenet
        target = self.prenet(target)    # (B, d_model, T=150)

        # global conditionの結合後、カーネル数1の畳み込み

        # positional encoding
        target = target.permute(0, -1, -2)  # (B, T, C)
        target = self.dropout(self.position_enc(target))

        target = self.layer_norm(target)
        dec_output = target

        dec_slf_attn_list = []
        dec_enc_attn_list = []

        # teacher forcing
        if training_method == "tf":
            # decoder layers
            for dec_layer in self.layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output, slf_attn_mask=target_mask, dec_enc_attn_mask=mask)
                dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        # scheduled sampling
        elif training_method == "ss":
            # decoder layers
            for dec_layer in self.layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output, slf_attn_mask=target_mask, dec_enc_attn_mask=mask)
                dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []
            
            for k in range(num_passes):
                # decoderからの出力とtargetをmixing_probに従って混合
                mixing_prob = torch.zeros_like(target) + mixing_prob
                judge = torch.bernoulli(mixing_prob)
                judge[:, :, :k] = 1     # t < kの要素は変更しない
                target = torch.where(judge == 1, target, dec_output)
                dec_output = target
                
                # decoder layers
                for dec_layer in self.layer_stack:
                    dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                        dec_output, enc_output, slf_attn_mask=target_mask, dec_enc_attn_mask=mask)
                    dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                    dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)
        dec_output = self.conv_o(dec_output)
        dec_output = dec_output.reshape(B, D, -1)   
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

    def inference(self, enc_output, data_len=None, prev=None, gc=None, return_attns=False):
        """
        reduction_factorにより、
        口唇動画のフレームの、reduction_factor倍のフレームを同時に出力する

        input
        enc_output : (B, T, C)

        return
        out : (B, C, T)
        """
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels
        
        # global conditionの結合
        
        # reshape for reduction factor
        if prev is not None:
            prev = prev.permute(0, -1, -2)
            prev = prev.reshape(B, -1, D * self.reduction_factor)
            prev = prev.permute(0, -1, -2)
        else:
            go_frame = torch.zeros(B, D * self.reduction_factor, 1)
            prev = go_frame

        max_decoder_time_steps = T

        # get target_mask
        # 推論時はいらないはず…
        mask = None
        target_mask = None

        # メインループ
        outs = []
        for _ in tqdm(range(max_decoder_time_steps)):
            # Prenet
            pre_out = self.prenet(prev)    # (B, C=d_model, T)
            
            # positional encoding
            pre_out = pre_out.permute(0, -1, -2)  # (B, T, C)
            dec_output = self.dropout(self.position_enc(pre_out))
            dec_output = self.layer_norm(dec_output)
            dec_slf_attn_list = []
            dec_enc_attn_list = []

            # decoder layers
            for dec_layer in self.layer_stack:
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output, slf_attn_mask=target_mask, dec_enc_attn_mask=mask)
                dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []

            dec_output = dec_output.permute(0, -1, -2)  # (B, C, T)
            dec_output = self.conv_o(dec_output)
            # dec_output = dec_output.reshape(B, D, -1)   

            # 出力を保持
            outs.append(dec_output.reshape(B, D, -1)[:, :, -2:])

            # 次のループへの入力
            # 前時刻までの全ての出力にしています
            prev = torch.cat((prev, dec_output[:, :, -1].unsqueeze(-1)), dim=2)
                
        # 各時刻の出力を結合して出力
        out = torch.cat(outs, dim=2)
        assert out.shape[-1] == T * 2
        return out


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
        conv.append(weight_norm(nn.Conv1d(inner_channels, out_channels, kernel_size=5, padding=5//2)))
        self.conv = conv

    def forward(self, x):
        return x + self.conv(x)



def main():
    # data_len
    data_len = [300, 300, 300, 300, 100, 100, 200, 200]
    data_len = torch.tensor(data_len)

    # 3DResnetからの出力
    batch = 8
    channels = 256
    t = 150
    prenet_out = torch.rand(batch, channels, t)
    # prenet_out = prenet_out.permute(0, 2, 1)    # (B, T, C)

    # 音響特徴量
    feature_channels = 80
    feature = torch.rand(batch, feature_channels, t*2)

    # transformer parameter
    n_layers = 6
    d_model = 256
    d_inner = d_model * 4   # 原論文と同様
    n_head = 8
    d_k = d_model // n_head
    d_v = d_model // n_head

    # mask
    target_mask = get_subsequent_mask(feature)
    print(target_mask)

if __name__ == "__main__":
    main()