from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_remake import make_pad_mask
from data_process.phoneme_encode import IGNORE_INDEX
from model.model_tts import Encoder, Attention, PreNet, PostNet, ZoneOutCell


class LipPreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        h = hidden_channels
        in_cs = [in_channels, h, h * 2, h * 4]
        out_cs = [h, h * 2, h * 4, h * 8]
        layers = []
        for i in range(len(in_cs)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_cs[i], in_cs[i], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(in_cs[i]),
                    nn.ReLU(),
                    nn.Conv2d(in_cs[i], out_cs[i], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_cs[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, H, W)
        """
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, dim=(2, 3))   # (B, C)
        return x


class LipOutLayer(nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout):
        super().__init__()
        h = hidden_channels
        in_cs = [h, h // 2, h // 4, h // 8]
        out_cs = [h // 2, h // 4, h // 8, out_channels]
        layers = []
        for i in range(len(in_cs)):
            if i != len(in_cs) - 1:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_cs[i], in_cs[i], kernel_size=3, padding=1),
                        nn.BatchNorm2d(in_cs[i]),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_cs[i], out_cs[i], kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(out_cs[i]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_cs[i], in_cs[i], kernel_size=3, padding=1),
                        nn.ConvTranspose2d(in_cs[i], out_cs[i], kernel_size=4, stride=2, padding=1),
                    )
                )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C)
        """
        x = x.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        x = F.interpolate(x, size=(3, 3), mode="nearest")   # (B, C, 3, 3)
        for layer in self.layers:
            x = layer(x)
        return x    # (B, C, H, W)


class Decoder(nn.Module):
    def __init__(
        self, enc_channels, dec_channels, atten_conv_channels, atten_conv_kernel_size, atten_hidden_channels,
        rnn_n_layers, prenet_hidden_channels, prenet_n_layers, out_channels, reduction_factor, dropout, use_gc, spk_emb_dim,
        lip_channels, lip_prenet_hidden_channels, lip_prenet_dropout, lip_out_hidden_channels, lip_out_dropout):
        super().__init__()
        self.enc_channels = enc_channels
        self.prenet_hidden_channels = prenet_hidden_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        self.lip_channels = lip_channels
        self.reduction_factor = reduction_factor

        self.attention = Attention(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conv_channels=atten_conv_channels,
            conv_kernel_size=atten_conv_kernel_size,
            hidden_channels=atten_hidden_channels,
        )

        self.prenet = PreNet(int(out_channels * reduction_factor), prenet_hidden_channels, prenet_n_layers)
        self.lip_prenet = LipPreNet(lip_channels, lip_prenet_hidden_channels, lip_prenet_dropout)
        self.prenet_out_concat_layer = nn.Linear(prenet_hidden_channels + int(lip_prenet_hidden_channels * 8), prenet_hidden_channels)

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
        self.lip_out_layer = nn.Sequential(
            nn.Linear(enc_channels + dec_channels, lip_out_hidden_channels),
            LipOutLayer(lip_out_hidden_channels, lip_channels, lip_out_dropout),
        )

    def _zero_state(self, hs, i):
        init_hs = hs.new_zeros(hs.size(0), self.dec_channels)
        return init_hs

    def forward(self, enc_output, text_len, feature_target=None, lip_target=None, spk_emb=None):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        feature_target : (B, C, T)
        lip_target : (B, C, H, W, T)
        spk_emb : (B, C)
        """
        # 音響特徴量のreduction factorに伴うreshape
        if feature_target is not None:
            B, C, T = feature_target.shape
            feature_target = feature_target.permute(0, 2, 1)
            feature_target = feature_target.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))
        else:
            B = enc_output.shape[0]
            C = self.out_channels

        # ループの上限
        if feature_target is not None:
            max_decoder_time_step = feature_target.shape[1]
        else:
            max_decoder_time_step = 1000

        mask = make_pad_mask(text_len, enc_output.shape[1]).squeeze(1)      # (B, T)

        # lstmの初期化
        h_list, c_list = [], []
        for i in range(len(self.lstm)):
            h_list.append(self._zero_state(enc_output, i))
            c_list.append(self._zero_state(enc_output, i))

        # 1フレーム目は0で初期化
        go_frame = enc_output.new_zeros(enc_output.size(0), int(self.out_channels * self.reduction_factor))
        prev_out = go_frame
        go_frame_lip = enc_output.new_zeros(enc_output.shape[0], self.lip_channels, 48, 48)
        prev_out_lip = go_frame_lip

        prev_att_w = None
        self.attention.reset()

        output_list = []
        lip_output_list = []
        logit_list = []
        att_w_list = []
        t = 0
        while True:
            att_c, att_w = self.attention(enc_output, text_len, h_list[0], prev_att_w, mask=mask)

            prenet_out = self.prenet(prev_out)      # (B, C)
            lip_prenet_out = self.lip_prenet(prev_out_lip)  # (B, C)
            prenet_out = self.prenet_out_concat_layer(torch.cat([prenet_out, lip_prenet_out], dim=1))   # (B, C)

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
            lip_output = self.lip_out_layer(hcs)    # (B, C, H, W)
            logit = self.prob_out_layer(hcs)    # (B, reduction_factor)

            output_list.append(output)
            lip_output_list.append(lip_output)
            logit_list.append(logit)
            att_w_list.append(att_w)

            if feature_target is not None:
                prev_out = feature_target[:, t, :]
                prev_out_lip = lip_target[..., t]
            else:
                prev_out = output
                prev_out_lip = lip_output

            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
            if t > max_decoder_time_step - 1:
                break
            if feature_target is None and (torch.sigmoid(logit) >= 0.5).any():
                break

        output = torch.cat(output_list, dim=1)  # (B, T * C)
        output = output.reshape(B, -1, C).permute(0, 2, 1)  # (B, C, T)
        lip_output = torch.stack(lip_output_list, dim=-1)   # (B, C, H, W, T)
        logit = torch.cat(logit_list, dim=-1)   # (B, T)
        att_w = torch.stack(att_w_list, dim=1)  # (B, T, C)
        return output, lip_output, logit, att_w


class LipPostNet(nn.Module):
    def __init__(self, out_channels, hidden_channels, n_layers, kernel_size, dropout):
        super().__init__()
        layers = []
        padding = (kernel_size - 1) // 2
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
        layers.append(nn.Conv3d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Tacotron2Face(nn.Module):
    def __init__(
        self, n_vocab, enc_hidden_channels, enc_conv_n_layers, enc_conv_kernel_size, enc_rnn_n_layers, enc_dropout,
        dec_channels, dec_atten_conv_channels, dec_atten_conv_kernel_size, dec_atten_hidden_channels, dec_rnn_n_layers, 
        dec_prenet_hidden_channels, dec_prenet_n_layers, out_channels, reduction_factor, dec_dropout,
        post_hidden_channels, post_n_layers, post_kernel_size, use_gc, spk_emb_dim,
        lip_channels, lip_prenet_hidden_channels, lip_prenet_dropout, lip_out_hidden_channels, lip_out_dropout,
        lip_post_hidden_channels, lip_post_n_layers, lip_post_kernel_size, lip_post_dropout):
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
            lip_channels=lip_channels,
            lip_prenet_hidden_channels=lip_prenet_hidden_channels,
            lip_prenet_dropout=lip_prenet_dropout,
            lip_out_hidden_channels=lip_out_hidden_channels,
            lip_out_dropout=lip_out_dropout,
        )
        self.postnet = PostNet(
            out_channels=out_channels,
            hidden_channels=post_hidden_channels,
            n_layers=post_n_layers,
            kernel_size=post_kernel_size,
        )
        self.lip_postnet = LipPostNet(
            out_channels=lip_channels,
            hidden_channels=lip_post_hidden_channels,
            n_layers=lip_post_n_layers,
            kernel_size=lip_post_kernel_size,
            dropout=lip_post_dropout,
        )

    def forward(self, text, text_len, feature_target=None, lip_target=None, spk_emb=None):
        """
        text : (B, T)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        enc_output = self.encoder(text, text_len, spk_emb)
        dec_output, dec_lip_output, logit, att_w = self.decoder(enc_output, text_len, feature_target, lip_target, spk_emb)
        output = self.postnet(dec_output)
        lip_output = self.lip_postnet(dec_lip_output)
        return dec_output, output, dec_lip_output, lip_output, logit, att_w


if __name__ == "__main__":
    net = LipOutLayer(
        hidden_channels=256,
        out_channels=1,
        dropout=0.1,
    )
    x = torch.rand(1, 256)
    out = net(x)