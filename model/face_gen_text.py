from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.transformer_remake import make_pad_mask, get_subsequent_mask, \
    posenc, MultiHeadAttention, PositionwiseFeedForward
from data_process.phoneme_encode import IGNORE_INDEX


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, mask):
        """
        enc_input : (B, T, C)
        """
        enc_output = self.attention(enc_input, enc_input, enc_input, mask)
        enc_output = self.fc(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.dec_self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.dec_enc_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_output, self_attention_mask, dec_enc_attention_mask):
        """
        dec_input : (B, T, C)
        enc_output : (B, T, C)
        """
        dec_output = self.dec_self_attention(dec_input, dec_input, dec_input, mask=self_attention_mask)
        dec_output = self.dec_enc_attention(dec_output, enc_output, enc_output, mask=dec_enc_attention_mask)
        dec_output = self.fc(dec_output)
        return dec_output


class Encoder(nn.Module):
    def __init__(self, n_vocab, n_layers, n_head, d_model, kernel_size, conv_n_layers, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4

        self.emb = nn.Embedding(n_vocab, d_model, padding_idx=IGNORE_INDEX)
        layers = []
        for i in range(conv_n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, text_len):
        """
        x : (B, T)
        text_len : (B,)
        """
        mask = make_pad_mask(text_len, x.shape[-1])
        x = self.emb(x).permute(0, 2, 1)    # (B, C, T)
        for layer in self.layers:
            x = layer(x)

        x = self.fc(x)
        x = x + posenc(x, device=x.device, start_index=0)
        x = x.permute(0, -1, -2)  # (B, T, C)
        x = self.layer_norm(x)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, mask)
        return x


class LipPreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, is_large):
        super().__init__()
        self.dropout = dropout
        h = hidden_channels
        in_cs = [in_channels, h, h * 2, h * 4]
        out_cs = [h, h * 2, h * 4, h * 8]

        layers = []
        for i in range(len(in_cs)):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_cs[i], out_cs[i], kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_cs[i]),
                    nn.ReLU(),
                )
            )
        self.layers = nn.ModuleList(layers)
        if is_large:
            self.pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout)

            if hasattr(self, "pool"):
                x = self.pool(x)

        x = torch.mean(x, dim=(2, 3))   # (B, C, T)
        return x


class LipOutLayer(nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout, is_large):
        super().__init__()
        h = hidden_channels
        in_cs = [h, h // 2, h // 4, h // 8]
        out_cs = [h // 2, h // 4, h // 8, out_channels]
        if is_large:
            self.expand_layer = nn.Sequential(
                nn.ConvTranspose3d(in_cs[0], in_cs[0], kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
                nn.BatchNorm3d(in_cs[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        layers = []
        for i in range(len(in_cs) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_cs[i], out_cs[i], kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_cs[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.ConvTranspose3d(in_cs[-1], out_cs[-1], kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0))

    def forward(self, x):
        """
        x : (B, T, C)
        """
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = x.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        x = F.interpolate(x, size=(3, 3, x.shape[-1]), mode="nearest")   # (B, C, 3, 3, T)
        if hasattr(self, "expand_layer"):
            x = self.expand_layer(x)

        for layer in self.layers:
            x = layer(x)
        x = self.last_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, n_layers, n_head, d_model, lip_channels, prenet_hidden_channels, 
        prenet_dropout, out_dropout, is_large, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4

        self.prenet = LipPreNet(lip_channels, prenet_hidden_channels, prenet_dropout, is_large)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])

        self.out_layer = LipOutLayer(d_model, lip_channels, out_dropout, is_large)
        self.prob_out_layer = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_output, text_len, lip_len, prev):
        """
        enc_output : (B, T, C)
        text_len, lip_len : (B,)
        prev : (B, C, H, W, T)
        """
        self_attention_mask = make_pad_mask(lip_len, prev.shape[-1]) | get_subsequent_mask(prev)
        dec_enc_attention_mask = make_pad_mask(text_len, enc_output.shape[1])
        prev = self.prenet(prev)
        prev = prev + posenc(prev, device=prev.device, start_index=0)
        prev = self.layer_norm(prev.permute(0, -1, -2))     # (B, T, C)

        for dec_layer in self.dec_layers:
            prev = dec_layer(prev, enc_output, self_attention_mask, dec_enc_attention_mask)

        dec_output = self.out_layer(prev)
        logit = self.prob_out_layer(prev).reshape(enc_output.shape[0], -1)
        return dec_output, logit


class PostNet(nn.Module):
    def __init__(self, out_channels, hidden_channels, n_layers, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = []
        for i in range(n_layers - 1):
            in_channels = out_channels if i == 0 else hidden_channels
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm3d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Conv3d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.last_layer(x)
        return x

    
class TransformerFaceGenerator(nn.Module):
    def __init__(
        self, n_vocab, enc_n_layers, enc_n_head, enc_d_model, enc_conv_kernel_size, enc_conv_n_layers,
        dec_n_layers, dec_n_head, dec_d_model, lip_channels, prenet_hidden_channels, prenet_dropout,
        dec_out_dropout, post_hidden_channels, post_n_layers, post_kernel_size, post_dropout, is_large):
        super().__init__()
        self.lip_channels = lip_channels
        self.encoder = Encoder(
            n_vocab=n_vocab,
            n_layers=enc_n_layers,
            n_head=enc_n_head,
            d_model=enc_d_model,
            kernel_size=enc_conv_kernel_size,
            conv_n_layers=enc_conv_n_layers,
        )
        self.decoder = Decoder(
            n_layers=dec_n_layers,
            n_head=dec_n_head,
            d_model=dec_d_model,
            lip_channels=lip_channels,
            prenet_hidden_channels=prenet_hidden_channels,
            prenet_dropout=prenet_dropout,
            out_dropout=dec_out_dropout,
            is_large=is_large,
        )
        self.postnet = PostNet(
            out_channels=lip_channels,
            hidden_channels=post_hidden_channels,
            n_layers=post_n_layers,
            kernel_size=post_kernel_size,
            dropout=post_dropout,
        )

    @autocast()
    def forward(self, text, text_len, lip_len, prev):
        """
        text : (B, T)
        text_len, lip_len : (B,)
        prev : (B, C, H, W, T)
        """
        enc_output = self.encoder(text, text_len)
        prev = F.pad(prev, (1, 0), mode="constant")[..., :-1]
        dec_output, logit = self.decoder(enc_output, text_len, lip_len, prev)
        output = self.postnet(dec_output)
        return dec_output, output, logit

    def inference(self, text, text_len, n_max_loop=1000):
        """
        text : (B, T)
        text_len : (B,)
        spk_emb : (B, C)
        """
        enc_output = self.encoder(text, text_len)
        t = 0
        prev = torch.zeros(enc_output.shape[0], self.lip_channels, 48, 48, 1).to(device=text.device, dtype=torch.float32)
        dec_output_list = []
        dec_output_list.append(prev)

        while True:
            lip_len = torch.tensor(prev.shape[-1]).unsqueeze(0)     # (B,)
            dec_output, logit = self.decoder(enc_output, text_len, lip_len, prev)
            dec_output_list.append(dec_output[..., -1:])
            t += 1
            if (torch.sigmoid(logit[:, -1:]) >= 0.5).any():
                break
            if t > n_max_loop:
                break
            prev = torch.cat(dec_output_list, dim=-1)

        dec_output = torch.cat(dec_output_list[1:], dim=-1)
        output = self.postnet(dec_output)
        return dec_output, output
