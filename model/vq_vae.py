import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .transformer_remake import EncoderLayer, make_pad_mask, posenc
from .taco import PreNet

class VQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = Encoder()
        self.vq = VectorQuantizer(num_embeddings=512, embedding_dim=256, commitment_cost=0.25)
        self.decoder = Decoder()
        
    def forward(self, feature, data_len):
       
        enc, tmp_data_len = self.encoder(feature, data_len)
        loss, vq, _, _ = self.vq(enc)
        
        out = self.decoder(vq, tmp_data_len)
        
        all_out = {}
        all_out['output'] = out
        all_out['vq_loss'] = loss

        return all_out
        
class VQVAE_Content_ResTC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb_dim = 128
        
        self.content_enc = ContentEncoder(
            in_channels=80,
            out_channels=emb_dim,
            n_attn_layer=2,
            n_head=4,
            reduction_factor=1,
            norm_type='bn',
        )
        self.vq = VectorQuantizer(num_embeddings=512, embedding_dim=emb_dim, commitment_cost=0.25)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=emb_dim,
            out_channels=80,
            inner_channels=256,
            n_layers=3,
            kernel_size=5,
            dropout=0.5,
            feat_add_channels=80, 
            feat_add_layers=80,
            use_feat_add=False,
            phoneme_classes=53,
            use_phoneme=False,
            n_attn_layer=1,
            n_head=4,
            d_model=emb_dim,
            reduction_factor=1,
            use_attention=False,
            compress_rate=2,
            upsample_method='conv'
        )
        
    def forward(self, feature, data_len):
        enc_output = self.content_enc(feature, data_len)
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)
        output = self.decoder(vq, data_len)
        
        all_out = {}
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding

        return all_out

class VQVAE_Content_AR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.content_enc = ContentEncoder(
            in_channels=80,
            out_channels=256,
            n_attn_layer=2,
            n_head=4,
            reduction_factor=1,
            norm_type='bn',
        )
        self.vq = VectorQuantizer(num_embeddings=512, embedding_dim=256, commitment_cost=0.25)

        # decoder
        self.decoder = ARDecoder()
        
    def forward(self, feature, data_len):
        enc_output = self.content_enc(feature, data_len)
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)
        output = self.decoder(enc_output, feature)
        
        all_out = {}
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding

        return all_out
    
class Encoder(nn.Module):
    def __init__(self, in_channels=80, out_channels=256):
        super().__init__()
        o = out_channels
        in_cs = [in_channels, out_channels, out_channels]
        out_cs = [o, o, o]
        self.stride = [1, 1, 2]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
            ) for in_c, out_c, s in zip(in_cs, out_cs, self.stride)
        ])
        
        self.lstm = nn.LSTM(
            out_channels, out_channels // 2, num_layers=2, batch_first=True, bidirectional=True
        )

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        out : (B, C)
        """
        out = x
        for layer in self.conv_layers:
            out = layer(out)
            
        for stride in self.stride:
            data_len //= stride
      
        x = out.permute(0, 2, 1)      # (B, T, C)

        
        seq_len_orig = x.shape[1]
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]
        
        if x.shape[1] < seq_len_orig:
            zero_pad = torch.zeros(x.shape[0], seq_len_orig - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)
            
        return out, data_len
    
class Decoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        o = 512
        in_cs = [in_channels, in_channels, in_channels]
        out_cs = [o, o, o]

        self.lstm_layers = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(
                        in_channels, in_channels//2, num_layers=2, batch_first=True, bidirectional=True
                        )
            )
        ])
        
        self.feat_out_layer = nn.Linear(in_channels, int(80 * 2), bias=False)

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        out : (B, C)
        """

        x = x.permute(0, 2, 1)
        seq_len_orig = x.shape[1]
        for layer in self.lstm_layers:
            x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = layer(x)
            x = pad_packed_sequence(x, batch_first=True)[0]
    
        x = self.feat_out_layer(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 80, -1)
       
        return x

class ResTCDecoder(nn.Module):
    """
    残差結合を取り入れたdecoder
    """
    def __init__(
        self, cond_channels, out_channels, inner_channels, n_layers, kernel_size, dropout, 
        feat_add_channels, feat_add_layers, use_feat_add, phoneme_classes, use_phoneme, 
        n_attn_layer, n_head, d_model, reduction_factor,  use_attention, compress_rate, upsample_method):
        super().__init__()
        self.use_phoneme = use_phoneme
        self.compress_rate = compress_rate
        self.use_attention = use_attention
        
        self.upsample_method = upsample_method 

        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose1d(cond_channels, inner_channels, kernel_size=compress_rate, stride=compress_rate),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.interp_layer = nn.Conv1d(cond_channels, inner_channels, kernel_size=1)

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
            ResBlock(inner_channels, inner_channels, kernel_size) for _ in range(n_layers)
        )

        self.out_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

    def forward(self, enc_output, data_len=None):
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
        
        # attention
        if self.use_attention:
            out = self.pre_attention_layer(out)
            out = self.attention(out, data_len, layer="dec")   # (B, T, C)
            out = self.post_attention_layer(out.permute(0, 2, 1))   # (B, C, T)

        for layer in self.conv_layers:
            out = layer(out)

        out = self.out_layer(out)
        return out
    
class ContentEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_attn_layer, n_head, reduction_factor, norm_type):
        super().__init__()
        assert out_channels % n_head == 0
        
        self.first_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_layers = nn.ModuleList([
            ResBlock(out_channels, out_channels, kernel_size=3, norm_type=norm_type),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            NormLayer1D(out_channels, norm_type),
            nn.ReLU(),
            ResBlock(out_channels, out_channels, kernel_size=3, norm_type=norm_type),
        ])

        self.attention = EncoderContent(
            n_layers=n_attn_layer, 
            n_head=n_head, 
            d_model=out_channels, 
            reduction_factor=reduction_factor,  
        )

    def forward(self, x, data_len=None):
        """
        x : (B, C, T)
        out : (B, T, C)
        """
        out = self.first_conv(x)
        for layer in self.conv_layers:
            out = layer(out)
        
        out = self.attention(out, data_len)
        return out
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
    
        """_summary_
        text_lenによる長さを考慮した損失計算

        Args:
            quantized (_type_): _description_
            inputs (_type_): _description_
            inputs_len (_type_): _description_
        """

    def forward(self, inputs, data_len):
        # convert inputs from BCHW -> BHWC
        data_len = data_len / 2
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = self.calc_mse(quantized.detach(), inputs, data_len)
        q_latent_loss = self.calc_mse(quantized, inputs.detach(), data_len)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings
    
    def calc_mse(self, output, target, data_len):
        def make_pad_mask_tts(lengths):
            """
            口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
            マスクする場所をTrue
            """
            # この後の処理でリストになるので先にdeviceを取得しておく
            device = lengths.device

            if not isinstance(lengths, list):
                lengths = lengths.tolist()
            bs = int(len(lengths))

            max_len = int(max(lengths))

            seq_range = torch.arange(0, max_len, dtype=torch.int64)
            seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
            seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
            mask = seq_range_expand >= seq_length_expand     
            return mask.unsqueeze(1).to(device=device)  # (B, 1, T)
         
        tmp_mask = make_pad_mask_tts(data_len)
        mask = tmp_mask.permute(0, 2, 1)
        mask = mask.repeat(1, 1, output.shape[-1])

        # 二乗誤差を計算
        loss = (output - target)**2

        # maskがTrueのところは0にして平均を取る
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=2)  # (B, T)

        # maskしていないところ全体で平均
        mask = tmp_mask.squeeze(1)  # (B, T)
        n_loss = torch.where(mask == 0, torch.ones_like(mask).to(torch.float32), torch.zeros_like(mask).to(torch.float32))
        mse_loss = torch.sum(loss) / torch.sum(n_loss)

        return mse_loss

    
class NormLayer1D(nn.Module):
    def __init__(self, in_channels, norm_type):
        super().__init__()
        self.norm_type = norm_type
        self.b_n = nn.BatchNorm1d(in_channels)
        self.i_n = nn.InstanceNorm1d(in_channels)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        if self.norm_type == "bn":
            out = self.b_n(x)
        elif self.norm_type == "in":
            out = self.i_n(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_type='bn'):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            NormLayer1D(out_channels, norm_type),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            NormLayer1D(out_channels, norm_type),
            nn.ReLU(),
        )

    def forward(self, x):
        res = x
        out = self.layers(x)
        out = out + res
        return out

class ARDecoder(nn.Module):
    def __init__(
        self, enc_channels=256, dec_channels=1024,
        rnn_n_layers=2, prenet_hidden_channels=256, prenet_n_layers=2, out_channels=80, reduction_factor=2, dropout=0.1):
        super().__init__()
        self.enc_channels = enc_channels
        self.prenet_hidden_channels = prenet_hidden_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        self.reduction_factor = reduction_factor


        self.prenet = PreNet(int(out_channels * reduction_factor), prenet_hidden_channels, prenet_n_layers)

        lstm = []
        for i in range(rnn_n_layers):
            lstm.append(
                ZoneOutCell(
                    nn.LSTMCell(
                        enc_channels + prenet_hidden_channels if i == 0 else dec_channels,
                        dec_channels,
                    ),
                    zoneout=dropout
                )
            )
        self.lstm = nn.ModuleList(lstm)
        self.feat_out_layer = nn.Linear(enc_channels + dec_channels, int(out_channels * reduction_factor), bias=False)

    def _zero_state(self, hs, i):
        init_hs = hs.new_zeros(hs.size(0), self.dec_channels)
        return init_hs

    def forward(self, enc_output, feature_target=None):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        
        training_method = "tf"
        
        #print(f'text len: {text_len}')
        breakpoint()
        if feature_target is not None:
            B, C, T = feature_target.shape
            feature_target = feature_target.permute(0, 2, 1)
            feature_target = feature_target.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))
        else:
            B = enc_output.shape[0]
            C = self.out_channels

        h_list, c_list = [], []
        for i in range(len(self.lstm)):
            h_list.append(self._zero_state(enc_output, i))
            c_list.append(self._zero_state(enc_output, i))

        go_frame = enc_output.new_zeros(enc_output.size(0), int(self.out_channels * self.reduction_factor))
        prev_out = go_frame

        output_list = []
        logit_list = []
        att_w_list = []
        att_c_list = []
        t = 0

        breakpoint()
        for t in range(enc_output.shape[1]):

            prenet_out = self.prenet(prev_out)      # (B, C)

            xs = torch.cat([att_c, prenet_out], dim=1)      # (B, C)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )

            hcs = torch.cat([h_list[-1], att_c], dim=1)     # (B, C)
            output = self.feat_out_layer(hcs)   # (B, C)
            logit = self.prob_out_layer(hcs)    # (B, reduction_factor)
            output_list.append(output)
            logit_list.append(logit)

            # if feature_target is not None:
            #     prev_out = feature_target[:, t, :]
            # else:
            #     prev_out = output
            if feature_target is not None:
                if training_method == "tf":
                    prev_out = feature_target[:, t, :]

                elif training_method == "ss":
                    """
                    mixing_prob = 1 : teacher forcing
                    mixing_prob = 0 : using decoder prediction completely
                    """
                    judge = torch.bernoulli(torch.tensor(mixing_prob))
                    if judge:
                        prev_out = feature_target[:, t, :]
                    else:
                        #prev_out = output.clone().detach()
                        prev_out = output.clone().detach()
            else:
                prev_out = output

        output = torch.cat(output_list, dim=1)  # (B, T, C)
        output = output.reshape(B, -1, C).permute(0, 2, 1)  # (B, C, T)
        
        return output

class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h
        
class EncoderContent(nn.Module):
    def __init__(self, n_layers, n_head, d_model, reduction_factor, dropout=0.1):
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

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, data_len=None, layer="enc"):
        """
        x : (B, C, T)
        enc_output : (B, T, C)
        """
        B, C, T = x.shape

        if data_len is not None:
            if layer == "enc":
                data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
            elif layer == "dec":
                pass
            max_len = T
            mask = make_pad_mask(data_len, max_len)
            
        else:
            mask = None

        x = self.dropout(x)
        x = x + self.alpha * posenc(x, device=x.device, start_index=0)
        x = x.permute(0, -1, -2)  # (B, T, C)
        enc_output = self.layer_norm(x)

        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, mask)
        return enc_output