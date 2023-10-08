import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from nar_decoder import ResTCDecoder, LinearDecoder
from rnn import GRUEncoder
from avhubert import Config, MyAVHubertModel, TransformerEncoder
from transformer_remake import PositionalEncoding


def load_avhubert_torch(
    cfg,
    ckpt_path,
    model_size,
    load_pretrained_weight,
    layer_loaded,
):
    if model_size == 'base':
        avhubert = MyAVHubertModel(cfg.base)
    elif model_size == 'large':
        avhubert = MyAVHubertModel(cfg.large)

    if load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))[str('avhubert')]
        avhubert_dict = avhubert.state_dict()

        # 読み込むパラメータを選択
        if layer_loaded == 'transformer':
            required_state_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('encoder.')}
        elif layer_loaded == 'resnet':
            required_state_dict = {k: v for k, v in pretrained_dict.items() if not (k.startswith('encoder.'))}
        elif layer_loaded == 'all':
            required_state_dict = pretrained_dict

        # state_dictは辞書なので、読み込みたいkey（layer）だけ重みをupdate
        avhubert_dict.update(required_state_dict)
        avhubert.load_state_dict(avhubert_dict)

    return avhubert


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        pos_enc_max_len,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=pos_enc_max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * 4),
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, padding_mask):
        '''
        x : (B, T, C)
        padding_mask : (B, T)
        '''
        x = x.permute(1, 0, 2)  # (T, B, C)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)  # (B, T, C)
        return x


class Lip2SP_NAR_AVHubert(nn.Module):
    def __init__(
        self,
        avhubert_config,
        avhubert_ckpt_path,
        avhubert_model_size,
        avhubert_return_res_output,
        load_avhubert_pretrained_weight,
        avhubert_layer_loaded,
        which_encoder,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
        which_decoder,
        out_channels,
        dec_n_layers,
        dec_kernel_size,
        dec_dropout,
        use_spk_emb,
        spk_emb_dim,
        dec_args,
        transformer_decoder_num_layers,
        pos_enc_max_len,
    ):
        super().__init__()
        self.avhubert_return_res_output = avhubert_return_res_output
        self.which_decoder = which_decoder
        self.avhubert = load_avhubert_torch(
            cfg=avhubert_config,
            ckpt_path=avhubert_ckpt_path,
            model_size=avhubert_model_size,
            load_pretrained_weight=load_avhubert_pretrained_weight,
            layer_loaded=avhubert_layer_loaded,
        )
        inner_channels = self.avhubert.encoder_embed_dim

        if use_spk_emb:
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        if which_decoder == 'restc':
            self.decoder = ResTCDecoder(
                cond_channels=inner_channels,
                out_channels=out_channels,
                inner_channels=inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
                reduction_factor=reduction_factor,
            )
        elif which_decoder == 'linear':
            self.decoder = LinearDecoder(
                in_channels=inner_channels,
                out_channels=out_channels,
                reduction_factor=reduction_factor,
            )
        elif which_decoder == 'transformer':
            self.decoder = TransformerDecoder(
                d_model=inner_channels,
                nhead=inner_channels // 64,
                num_layers=transformer_decoder_num_layers,
                pos_enc_max_len=pos_enc_max_len,
            )
            self.decoder_conv = ResTCDecoder(
                cond_channels=inner_channels,
                out_channels=out_channels,
                inner_channels=inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
                reduction_factor=reduction_factor,
            )
        elif which_decoder == 'gru':
            self.decoder = GRUEncoder(
                hidden_channels=inner_channels,
                n_layers=rnn_n_layers,
                dropout=rnn_dropout,
                reduction_factor=reduction_factor,
                which_norm=rnn_which_norm,
            )
            self.decoder_conv = ResTCDecoder(
                cond_channels=inner_channels,
                out_channels=out_channels,
                inner_channels=inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
                reduction_factor=reduction_factor,
            )

    def forward(
        self,
        lip,
        audio,
        lip_len,
        spk_emb,
    ):
        '''
        lip : (B, C, H, W, T)
        audio : (B, C, T)
        lip_len : (B,)
        spk_emb : (B, C)
        '''
        if lip is not None:
            lip = lip.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
            padding_mask_avhubert = torch.zeros(lip.shape[0], lip.shape[2]).to(device=lip.device, dtype=torch.bool)     # (B, T)
        elif audio is not None:
            padding_mask_avhubert = torch.zeros(audio.shape[0], audio.shape[2]).to(device=audio.device, dtype=torch.bool)     # (B, T)

        # padding部分がTrue
        for i, l in enumerate(lip_len):
            padding_mask_avhubert[i, l:] = True
        
        x = self.avhubert(
            video=lip,
            audio=audio, 
            return_res_output=self.avhubert_return_res_output,
            padding_mask=padding_mask_avhubert, 
        )   # (B, T, C)

        if self.avhubert_return_res_output:
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = self.encoder(x, lip_len)     # (B, T, C)

        if hasattr(self, 'spk_emb_layer'):
            x = x.permute(0, 2, 1)
            spk_emb = spk_emb.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])
            x = torch.cat([x, spk_emb], dim=1)
            x = self.spk_emb_layer(x)
            x = x.permute(0, 2, 1)

        if self.which_decoder == 'transformer':
            x = self.decoder(x, padding_mask=padding_mask_avhubert)     # (B, T, C)
            x = self.decoder_conv(x)
        elif self.which_decoder == 'gru':
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = self.decoder(x, lip_len)    # (B, T, C)
            x = self.decoder_conv(x)
        else:
            x = self.decoder(x)
        return x, None, None