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
    def __init__(self, n_vocab, n_layers, n_head, d_model, kernel_size, conv_n_layers, reduction_factor, which_norm, conv_dropout, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4
        self.reduction_factor = reduction_factor
        self.which_norm = which_norm

        self.emb = nn.Embedding(n_vocab, d_model, padding_idx=IGNORE_INDEX)
        convs = []
        norms = []
        for i in range(conv_n_layers):
            convs.append(nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2))
            if which_norm == "ln":
                norms.append(nn.LayerNorm(d_model, eps=1e-6))
            elif which_norm == "bn":
                norms.append(nn.BatchNorm1d(d_model))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(conv_dropout)
        self.fc = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_dropout = nn.Dropout(0.1)

    def forward(self, x, text_len, spk_emb=None):
        """
        x : (B, T)
        text_len : (B,)
        """
        mask = make_pad_mask(text_len, x.shape[-1])
        x = self.emb(x).permute(0, 2, 1)     # (B, C, T)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            if self.which_norm == "ln":
                x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            elif self.which_norm == "bn":
                x = norm(x)
            x = self.dropout(F.relu(x))

        x = self.fc(x)
        x = x + posenc(x, device=x.device, start_index=0)
        x = x.permute(0, -1, -2)  # (B, T, C)
        # x = self.layer_norm(x)
        x = self.pos_dropout(x)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, mask)
        return x


class PreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, dropout):
        super().__init__()
        self.dropout = dropout
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels if i == 0 else hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, T, C)
        """
        for layer in self.layers:
            # x = F.dropout(layer(x), p=self.dropout)
            x = layer(x)
        return x


class LipPreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, which_norm):
        super().__init__()
        self.dropout = dropout
        self.which_norm = which_norm
        h = hidden_channels
        in_cs = [in_channels, h, h * 2, h * 4]
        out_cs = [h, h * 2, h * 4, h * 8]
        shapes = [24, 12, 6, 3]
        self.n_layers = len(in_cs)

        convs = []
        norms = []
        for i in range(len(in_cs)):
            convs.append(nn.Conv3d(in_cs[i], out_cs[i], kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0)))
            if which_norm == "ln":
                norms.append(nn.LayerNorm([shapes[i], shapes[i], out_cs[i]], eps=1e-6))
            elif which_norm == "bn":
                norms.append(nn.BatchNorm3d(out_cs[i]))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            # if self.which_norm == "ln":
            #     x = norm(x.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)
            # elif self.which_norm == "bn":
            #     x = norm(x)
            # x = F.dropout(F.relu(x), p=self.dropout)
            x = self.dropout(F.relu(x))
        x = torch.mean(x, dim=(2, 3))   # (B, C, T)
        return x


class LipOutLayer(nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout, which_norm):
        super().__init__()
        self.which_norm = which_norm
        h = hidden_channels
        in_cs = [h, h // 2, h // 4, h // 8]
        out_cs = [h // 2, h // 4, h // 8, out_channels]
        shapes = [6, 12, 24]
        self.n_layers = len(in_cs) - 1

        tconvs = []
        norms = []
        for i in range(len(in_cs) - 1):
            tconvs.append(nn.ConvTranspose3d(in_cs[i], out_cs[i], kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0)))
            if which_norm == "ln":
                norms.append(nn.LayerNorm([shapes[i], shapes[i], out_cs[i]], eps=1e-6))
            elif which_norm == "bn":
                norms.append(nn.BatchNorm3d(out_cs[i]))
        self.tconvs = nn.ModuleList(tconvs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)
        self.last_layer = nn.ConvTranspose3d(in_cs[-1], out_cs[-1], kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0))

    def forward(self, x):
        """
        x : (B, T, C)
        """
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = x.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        x = F.interpolate(x, size=(3, 3, x.shape[-1]), mode="nearest")   # (B, C, 3, 3, T)

        for tconv, norm in zip(self.tconvs, self.norms):
            x = tconv(x)
            if self.which_norm == "ln":
                x = norm(x.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)
            elif self.which_norm == "bn":
                x = norm(x)
            x = self.dropout(F.relu(x))

        x = self.last_layer(x)
        return x    # (B, C, H, W, T)


class Decoder(nn.Module):
    def __init__(
        self, n_layers, n_head, d_model, feat_pre_hidden_channels, feat_pre_n_layers, out_channels, feat_prenet_dropout,
        lip_channels, lip_prenet_hidden_channels, lip_prenet_dropout, lip_out_hidden_channels, lip_out_dropout,
        which_norm, reduction_factor, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.lip_channels = lip_channels
        self.d_inner = d_model * 4

        self.prenet = PreNet(out_channels * reduction_factor, feat_pre_hidden_channels, feat_pre_n_layers, feat_prenet_dropout)
        self.lip_prenet = LipPreNet(lip_channels, lip_prenet_hidden_channels, lip_prenet_dropout, which_norm)
        self.prenet_concat_layer = nn.Conv1d(feat_pre_hidden_channels + int(lip_prenet_hidden_channels * 8), d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        self.feat_out_layer = nn.Linear(d_model, self.out_channels * self.reduction_factor)
        self.lip_out_layer = nn.Sequential(
            nn.Linear(d_model, lip_out_hidden_channels),
            LipOutLayer(lip_out_hidden_channels, lip_channels, lip_out_dropout, which_norm),
        )
        self.prob_out_layer = nn.Linear(d_model, reduction_factor)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_dropout = nn.Dropout(0.1)

    def forward(self, enc_output, text_len, lip_len, prev_feature, prev_lip, spk_emb=None):
        """
        enc_output : (B, T, C)
        text_len, lip_len : (B,)
        prev_feature : (B, C, T)
        prev_lip : (B, C, H, W, T)
        spk_emb : (B, C)
        """
        B, C, T = prev_feature.shape
        prev_feature = prev_feature.permute(0, 2, 1)
        prev_feature = prev_feature.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))  # (B, T, C)

        self_attention_mask = make_pad_mask(lip_len, prev_lip.shape[-1]) | get_subsequent_mask(prev_feature.permute(0, 2, 1))
        dec_enc_attention_mask = make_pad_mask(text_len, enc_output.shape[1])

        prev_feature = self.prenet(prev_feature).permute(0, 2, 1)    # (B, C, T)
        prev_lip = self.lip_prenet(prev_lip)    # (B, C, T)
        prev = self.prenet_concat_layer(torch.cat([prev_feature, prev_lip], dim=1))     # (B, C, T)

        prev = prev + posenc(prev, device=prev.device, start_index=0)
        # prev = self.layer_norm(prev.permute(0, -1, -2))     # (B, T, C)
        prev = self.pos_dropout(prev.permute(0, -1, -2))     # (B, T, C)

        for i, dec_layer in enumerate(self.dec_layers):
            prev = dec_layer(prev, enc_output, self_attention_mask, dec_enc_attention_mask)

        dec_feat_output = self.feat_out_layer(prev)     # (B, T, C)
        dec_feat_output = dec_feat_output.reshape(enc_output.shape[0], -1, self.out_channels)
        dec_feat_output = dec_feat_output.permute(0, 2, 1)  # (B, C, T)
        dec_lip_output = self.lip_out_layer(prev)   # (B, C, H, W, T)
        logit = self.prob_out_layer(prev)   # (B, T, reduction_factor)
        logit = logit.reshape(enc_output.shape[0], -1)  # (B, T)
        return dec_feat_output, dec_lip_output, logit


class PostNet(nn.Module):
    def __init__(self, out_channels, hidden_channels, n_layers, kernel_size, which_norm, dropout):
        super().__init__()
        self.which_norm = which_norm
        padding = (kernel_size - 1) // 2
        self.n_layers = n_layers - 1
        convs = []
        norms = []

        for i in range(n_layers - 1):
            in_channels = out_channels if i == 0 else hidden_channels
            convs.append(nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False))
            if which_norm == "ln":
                norms.append(nn.LayerNorm(hidden_channels, eps=1e-6))
            elif which_norm == "bn":
                norms.append(nn.BatchNorm1d(hidden_channels))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)
        self.last_layer = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            if self.which_norm == "ln":
                x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            elif self.which_norm == "bn":
                x = norm(x)
            x = self.dropout(torch.tanh(x))
        x = self.last_layer(x)
        return x


class LipPostNet(nn.Module):
    def __init__(self, out_channels, hidden_channels, n_layers, kernel_size, which_norm, dropout):
        super().__init__()
        self.which_norm = which_norm
        padding = (kernel_size - 1) // 2
        self.n_layers = n_layers - 1
        convs = []
        norms = []

        for i in range(n_layers - 1):
            in_channels = out_channels if i == 0 else hidden_channels
            convs.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding))
            if which_norm == "ln":
                norms.append(nn.LayerNorm([48, 48, hidden_channels], eps=1e-6))
            elif which_norm == "bn":
                norms.append(nn.BatchNorm3d(hidden_channels))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)
        self.last_layer = nn.Conv3d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            if self.which_norm == "ln":
                x = norm(x.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)
            elif self.which_norm == "bn":
                x = norm(x)
            x = self.dropout(F.relu(x))
        x = self.last_layer(x)
        return x


class TransformerFaceSpeechSynthesizer(nn.Module):
    def __init__(
        self, n_vocab, enc_n_layers, enc_n_head, enc_d_model, reduction_factor,
        enc_conv_kernel_size, enc_conv_n_layers, enc_conv_dropout,
        dec_n_layers, dec_n_head, dec_d_model, feat_pre_hidden_channels, feat_pre_n_layers, feat_prenet_dropout,
        out_channels, lip_channels, lip_prenet_hidden_channels, lip_prenet_dropout,
        lip_out_hidden_channels, lip_out_dropout, feat_post_hidden_channels, feat_post_n_layers,
        feat_post_kernel_size, feat_post_dropout,
        lip_post_hidden_channels, lip_post_n_layers, lip_post_kernel_size, lip_post_dropout, which_norm):
        super().__init__()
        self.out_channels = out_channels
        self.lip_channels = lip_channels
        self.reduction_factor = reduction_factor
        self.encoder = Encoder(
            n_vocab=n_vocab,
            n_layers=enc_n_layers,
            n_head=enc_n_head,
            d_model=enc_d_model,
            kernel_size=enc_conv_kernel_size,
            conv_n_layers=enc_conv_n_layers,
            which_norm=which_norm,
            reduction_factor=reduction_factor,
            conv_dropout=enc_conv_dropout,
        )
        self.decoder = Decoder(
            n_layers=dec_n_layers,
            n_head=dec_n_head,
            d_model=dec_d_model,
            feat_pre_hidden_channels=feat_pre_hidden_channels,
            feat_pre_n_layers=feat_pre_n_layers,
            feat_prenet_dropout=feat_prenet_dropout,
            out_channels=out_channels,
            lip_channels=lip_channels,
            lip_prenet_hidden_channels=lip_prenet_hidden_channels,
            lip_prenet_dropout=lip_prenet_dropout,
            lip_out_hidden_channels=lip_out_hidden_channels,
            lip_out_dropout=lip_out_dropout,
            which_norm=which_norm,
            reduction_factor=reduction_factor,
        )
        self.feat_postnet = PostNet(
            out_channels=out_channels,
            hidden_channels=feat_post_hidden_channels,
            n_layers=feat_post_n_layers,
            kernel_size=feat_post_kernel_size,
            which_norm=which_norm,
            dropout=feat_post_dropout,
        )
        self.lip_postnet = LipPostNet(
            out_channels=lip_channels,
            hidden_channels=lip_post_hidden_channels,
            n_layers=lip_post_n_layers,
            kernel_size=lip_post_kernel_size,
            which_norm=which_norm,
            dropout=lip_post_dropout,
        )

    @autocast()
    def forward(self, text, text_len, lip_len, prev_feature=None, prev_lip=None, spk_emb=None):
        """
        text : (B, T)
        text_len, lip_len : (B,)
        prev_feature : (B, C, T)
        prev_lip : (B, C, H, W, T)
        spk_emb : (B, C)
        """
        enc_output = self.encoder(text, text_len, spk_emb)

        # シフト
        prev_feature = F.pad(prev_feature, (self.reduction_factor, 0), mode="constant")[..., :-self.reduction_factor]
        prev_lip = F.pad(prev_lip, (1, 0), mode="constant")[..., :-1]

        dec_feat_output, dec_lip_output, logit = self.decoder(enc_output, text_len, lip_len, prev_feature, prev_lip, spk_emb)
        feat_output = self.feat_postnet(dec_feat_output)
        lip_output = self.lip_postnet(dec_lip_output)

        return dec_feat_output, dec_lip_output, feat_output, lip_output, logit

    def inference(self, text, text_len, lip_len, spk_emb, n_max_loop=300):
        """
        text : (B, T)
        text_len, lip_len : (B,)
        spk_emb : (B, C)
        """
        enc_output = self.encoder(text, text_len, spk_emb)
        t = 0
        prev_feature = torch.zeros(enc_output.shape[0], self.out_channels, self.reduction_factor).to(device=text.device, dtype=torch.float32)
        prev_lip = torch.zeros(enc_output.shape[0], self.lip_channels, 48, 48, 1).to(device=text.device, dtype=torch.float32)
        dec_feat_output_list = []
        dec_lip_output_list = []
        dec_feat_output_list.append(prev_feature)
        dec_lip_output_list.append(prev_lip)

        while True:
            dec_feat_output, dec_lip_output, logit = self.decoder(enc_output, text_len, lip_len, prev_feature, prev_lip, spk_emb)
            dec_feat_output_list.append(dec_feat_output[..., -self.reduction_factor:])
            dec_lip_output_list.append(dec_lip_output[..., -1:])

            t += 1
            if (torch.sigmoid(logit[:, -self.reduction_factor:]) >= 0.5).any():
                break
            if t > n_max_loop:
                break

            prev_feature = torch.cat(dec_feat_output_list, dim=-1)
            prev_lip = torch.cat(dec_lip_output_list, dim=-1)

        dec_feat_output = torch.cat(dec_feat_output_list[1:], dim=-1)
        dec_lip_output = torch.cat(dec_lip_output_list[1:], dim=-1)
        feat_output = self.feat_postnet(dec_feat_output)
        lip_output = self.lip_postnet(dec_lip_output)
        return dec_feat_output, dec_lip_output, feat_output, lip_output
