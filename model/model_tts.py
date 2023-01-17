from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.transformer_remake import make_pad_mask
from data_process.phoneme_encode import IGNORE_INDEX


class Encoder(nn.Module):
    def __init__(self, n_vocab, hidden_channels, conv_n_layers, conv_kernel_size, rnn_n_layers, dropout, use_gc, spk_emb_dim):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=IGNORE_INDEX)
        conv_layers = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_n_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=conv_kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lstm = nn.LSTM(
            hidden_channels, hidden_channels // 2, num_layers=rnn_n_layers, batch_first=True, bidirectional=True
        )

        if use_gc:
            self.spk_emb_layer = nn.Linear(hidden_channels + spk_emb_dim, hidden_channels)

    def forward(self, x, text_len, spk_emb=None):
        """
        x : (B, T)
        text_len : (B,)
        spk_emb : (B, C)
        """
        x = self.emb(x)     # (B, T, C)
        x = x.permute(0, 2, 1)      # (B, C, T)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)      # (B, T, C)

        x = pack_padded_sequence(x, text_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]

        if hasattr(self, "spk_emb_layer"):
            spk_emb = spk_emb.unsqueeze(1).expand(-1, x.shape[1], -1)   # (B, T, C)
            x = torch.cat([x, spk_emb], dim=-1)
            x = self.spk_emb_layer(x)

        return x    # (B, T, C)


class Attention(nn.Module):
    def __init__(self, enc_channels, dec_channels, conv_channels, conv_kernel_size, hidden_channels):
        super().__init__()
        self.fc_enc = nn.Linear(enc_channels, hidden_channels)
        self.fc_dec = nn.Linear(dec_channels, hidden_channels, bias=False)
        self.fc_att = nn.Linear(conv_channels, hidden_channels, bias=False)
        self.loc_conv = nn.Conv1d(1, conv_channels, conv_kernel_size, padding=(conv_kernel_size - 1) // 2, bias=False)
        self.w = nn.Linear(hidden_channels, 1)
        self.processed_memory = None

    def reset(self):
        self.processed_memory = None

    def forward(self, enc_output, text_len, dec_state, prev_att_w, mask=None):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        dec_state : (B, C)
        prev_att_w : (B, T)
        """
        if self.processed_memory is None:
            self.processed_memory = self.fc_enc(enc_output)     # (B, T, C)

        if prev_att_w is None:
            prev_att_w = 1.0 - make_pad_mask(text_len, enc_output.shape[1]).squeeze(1).to(torch.float32)   # (B, T)
            prev_att_w = prev_att_w / text_len.unsqueeze(1)

        att_conv = self.loc_conv(prev_att_w.unsqueeze(1))     # (B, C, T)
        att_conv = att_conv.permute(0, 2, 1)    # (B, T, C)
        att_conv = self.fc_att(att_conv)    # (B, T, C)

        dec_state = self.fc_dec(dec_state).unsqueeze(1)      # (B, 1, C)
        
        att_energy = self.w(torch.tanh(att_conv + self.processed_memory + dec_state))   # (B, T, 1)
        att_energy = att_energy.squeeze(-1)     # (B, T)

        if mask is not None:
            att_energy = att_energy.masked_fill(mask, torch.tensor(float('-inf')))

        att_w = F.softmax(att_energy, dim=1)    # (B, T)
        att_c = torch.sum(enc_output * att_w.unsqueeze(-1), dim=1)  # (B, C)
        return att_c, att_w


class PreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels if i == 0 else hidden_channels, hidden_channels),
                    nn.ReLU(),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C)
        """
        # dropoutは推論時も適用(同じ音素が連続するのを防ぐため,前時刻の出力にあえてランダム性を付加する)
        for layer in self.layers:
            x = F.dropout(layer(x))
        return x


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


class Decoder(nn.Module):
    def __init__(
        self, enc_channels, dec_channels, atten_conv_channels, atten_conv_kernel_size, atten_hidden_channels,
        rnn_n_layers, prenet_hidden_channels, prenet_n_layers, out_channels, reduction_factor, dropout, use_gc, spk_emb_dim):
        super().__init__()
        self.enc_channels = enc_channels
        self.prenet_hidden_channels = prenet_hidden_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        self.reduction_factor = reduction_factor

        self.attention = Attention(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conv_channels=atten_conv_channels,
            conv_kernel_size=atten_conv_kernel_size,
            hidden_channels=atten_hidden_channels,
        )

        self.prenet = PreNet(int(out_channels * reduction_factor), prenet_hidden_channels, prenet_n_layers)

        if use_gc:
            self.spk_emb_layer = nn.Linear(prenet_hidden_channels + spk_emb_dim, prenet_hidden_channels)
        
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
        self.prob_out_layer = nn.Linear(enc_channels + dec_channels, reduction_factor)

    def _zero_state(self, hs, i):
        init_hs = hs.new_zeros(hs.size(0), self.dec_channels)
        return init_hs

    def forward(self, enc_output, text_len, feature_target=None, spk_emb=None):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        if feature_target is not None:
            B, C, T = feature_target.shape
            feature_target = feature_target.permute(0, 2, 1)
            feature_target = feature_target.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))
        else:
            B = enc_output.shape[0]
            C = self.out_channels

        if feature_target is not None:
            max_decoder_time_step = feature_target.shape[1]
        else:
            max_decoder_time_step = 1000

        mask = make_pad_mask(text_len, enc_output.shape[1]).squeeze(1)      # (B, T)

        h_list, c_list = [], []
        for i in range(len(self.lstm)):
            h_list.append(self._zero_state(enc_output, i))
            c_list.append(self._zero_state(enc_output, i))

        go_frame = enc_output.new_zeros(enc_output.size(0), int(self.out_channels * self.reduction_factor))
        prev_out = go_frame

        prev_att_w = None
        self.attention.reset()

        output_list = []
        logit_list = []
        att_w_list = []
        t = 0
        while True:
            att_c, att_w = self.attention(enc_output, text_len, h_list[0], prev_att_w, mask=mask)

            prenet_out = self.prenet(prev_out)      # (B, C)

            if hasattr(self, "spk_emb_layer"):
                prenet_out = torch.cat([prenet_out, spk_emb], dim=-1)
                prenet_out = self.spk_emb_layer(prenet_out)

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
            att_w_list.append(att_w)

            if feature_target is not None:
                prev_out = feature_target[:, t, :]
            else:
                prev_out = output

            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
            if t > max_decoder_time_step - 1:
                break
            if feature_target is None and (torch.sigmoid(logit) >= 0.5).any():
                break

        output = torch.cat(output_list, dim=1)  # (B, T, C)
        output = output.reshape(B, -1, C).permute(0, 2, 1)  # (B, C, T)
        logit = torch.cat(logit_list, dim=-1)   # (B, T)
        att_w = torch.stack(att_w_list, dim=1)  # (B, T, C)
        return output, logit, att_w


class PostNet(nn.Module):
    def __init__(self, out_channels, hidden_channels, n_layers, kernel_size, dropout=0.5):
        super().__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        for i in range(n_layers - 1):
            in_channels = out_channels if i == 0 else hidden_channels
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Tacotron2(nn.Module):
    def __init__(
        self, n_vocab, enc_hidden_channels, enc_conv_n_layers, enc_conv_kernel_size, enc_rnn_n_layers, enc_dropout,
        dec_channels, dec_atten_conv_channels, dec_atten_conv_kernel_size, dec_atten_hidden_channels, dec_rnn_n_layers, 
        dec_prenet_hidden_channels, dec_prenet_n_layers, out_channels, reduction_factor, dec_dropout,
        post_hidden_channels, post_n_layers, post_kernel_size, use_gc, spk_emb_dim):
        super().__init__()
        self.encoder = Encoder(
            n_vocab=n_vocab,
            hidden_channels=enc_hidden_channels,
            conv_n_layers=enc_conv_n_layers,
            conv_kernel_size=enc_conv_kernel_size,
            rnn_n_layers=enc_rnn_n_layers,
            dropout=enc_dropout,
            use_gc=use_gc,
            spk_emb_dim=spk_emb_dim,
        )
        self.decoder = Decoder(
            enc_channels=enc_hidden_channels,
            dec_channels=dec_channels,
            atten_conv_channels=dec_atten_conv_channels,
            atten_conv_kernel_size=dec_atten_conv_kernel_size,
            atten_hidden_channels=dec_atten_hidden_channels,
            rnn_n_layers=dec_rnn_n_layers,
            prenet_hidden_channels=dec_prenet_hidden_channels,
            prenet_n_layers=dec_prenet_n_layers,
            out_channels=out_channels,
            reduction_factor=reduction_factor,
            dropout=dec_dropout,
            use_gc=use_gc,
            spk_emb_dim=spk_emb_dim,
        )
        self.postnet = PostNet(
            out_channels=out_channels,
            hidden_channels=post_hidden_channels,
            n_layers=post_n_layers,
            kernel_size=post_kernel_size,
        )

    def forward(self, text, text_len, feature_target=None, spk_emb=None):
        """
        text : (B, T)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        enc_output = self.encoder(text, text_len, spk_emb)
        dec_output, logit, att_w = self.decoder(enc_output, text_len, feature_target, spk_emb)
        output = self.postnet(dec_output)
        return dec_output, output, logit, att_w


if __name__ == "__main__":
    reduction_factor = 2
    B = 1
    C = 2
    T = 10
    feature_target = torch.rand(B, C, T)   # (B, C, T)
    print("goal")
    print(feature_target)
    feature_target = feature_target.permute(0, 2, 1)    # (B, T, C)
    print(feature_target)
    feature_target = feature_target.reshape(B, T // reduction_factor, -1)
    print(feature_target)

    output_list = []
    for i in range(feature_target.shape[1]):
        output_list.append(feature_target[:, i, :])

    output = torch.cat(output_list, dim=1)
    output = output.reshape(B, T, -1).permute(0, 2, 1)
    print(output)
    breakpoint()

