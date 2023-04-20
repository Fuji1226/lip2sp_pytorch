import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.model_tts import Encoder, Decoder, PostNet


class ReferenceEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_conv_layers, n_rnn_layers):
        super().__init__()
        convs = []
        for i in range(n_conv_layers):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = hidden_channels
            convs.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            ))
        self.convs = nn.ModuleList(convs)
        self.gru = nn.GRU(hidden_channels, hidden_channels // 2, num_layers=n_rnn_layers, batch_first=True, bidirectional=True)
        
    def forward(self, x, data_len):
        """
        Args:
            x (_type_): (B, C, T)
            data_len (_type_): (B,)
        """
        for layer in self.convs:
            x = layer(x)
        
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        x = pad_packed_sequence(x, batch_first=True)[0]     # (B, T, C)
        x = torch.mean(x, dim=1)   # (B, C)
        return x
    
    
class VAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_conv_layers, n_rnn_layers, z_dim, emb_dim):
        super().__init__()
        self.ref_enc = ReferenceEncoder(in_channels, hidden_channels, n_conv_layers, n_rnn_layers)
        self.fc_mu = nn.Linear(hidden_channels, z_dim)
        self.fc_var = nn.Linear(hidden_channels, z_dim)
        self.fc_emb = nn.Linear(z_dim, emb_dim)
        
    def reparam(self, mu, logvar):
        if self.training:
            std= torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
        else:
            z = mu
        return z
        
    def forward(self, x, data_len):
        x = self.ref_enc(x, data_len)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparam(mu, logvar)
        emb = self.fc_emb(z)
        return mu, logvar, z, emb
    
    
class Tacotron2VAE(nn.Module):
    def __init__(
        self, n_vocab, enc_hidden_channels, enc_conv_n_layers, enc_conv_kernel_size, enc_rnn_n_layers, enc_dropout,
        dec_channels, dec_atten_conv_channels, dec_atten_conv_kernel_size, dec_atten_hidden_channels, dec_rnn_n_layers, 
        dec_prenet_hidden_channels, dec_prenet_n_layers, out_channels, reduction_factor, dec_dropout,
        post_hidden_channels, post_n_layers, post_kernel_size, use_gc, spk_emb_dim,
        vae_hidden_channels, vae_n_conv_layers, vae_n_rnn_layers, z_dim):
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
        self.vae = VAE(
            in_channels=out_channels,
            hidden_channels=vae_hidden_channels,
            n_conv_layers=vae_n_conv_layers,
            n_rnn_layers=vae_n_rnn_layers,
            z_dim=z_dim,
            emb_dim=enc_hidden_channels,
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

    def forward(self, text, text_len, feature, feature_len, feature_target=None, spk_emb=None):
        """
        text : (B, T)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        enc_output = self.encoder(text, text_len, spk_emb)
        mu, logvar, z, emb = self.vae(feature, feature_len)
        emb = emb.unsqueeze(1).expand(-1, enc_output.shape[1], -1)  # (B, T, C)
        enc_output = enc_output + emb
        dec_output, logit, att_w = self.decoder(enc_output, text_len, feature_target, spk_emb)
        output = self.postnet(dec_output)
        return dec_output, output, logit, att_w, mu, logvar