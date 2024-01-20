import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from nar_decoder import ResTCDecoder, LinearDecoder
from rnn import GRUEncoder
from avhubert import MyAVHubertModel, ResEncoder
from transformer_remake import PositionalEncoding


def load_avhubert_torch(
    cfg,
    model_size,
    load_pretrained_weight,
    layer_loaded,
):
    if model_size == 'base':
        avhubert = MyAVHubertModel(cfg.base)
        ckpt_path = Path(cfg.base.ckpt_path).expanduser()
    elif model_size == 'large':
        avhubert = MyAVHubertModel(cfg.large)
        ckpt_path = Path(cfg.large.ckpt_path).expanduser()

    if load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))['avhubert']
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
        lip = lip.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        padding_mask_avhubert = torch.arange(lip.shape[2]).unsqueeze(0).expand(lip.shape[0], -1).to(device=lip.device)
        padding_mask_avhubert = padding_mask_avhubert > lip_len.unsqueeze(-1)
        
        x = self.avhubert(
            video=lip,
            audio=None, 
            return_res_output=self.avhubert_return_res_output,
            padding_mask=padding_mask_avhubert, 
        )   # (B, T, C)
        avhubert_feature = x

        if self.avhubert_return_res_output:
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = self.encoder(x, lip_len)     # (B, T, C)

        if hasattr(self, 'spk_emb_layer'):
            x = x.permute(0, 2, 1)  # (B, C, T)
            spk_emb = spk_emb.unsqueeze(-1).expand(spk_emb.shape[0], spk_emb.shape[1], x.shape[-1])
            x = torch.cat([x, spk_emb], dim=1)
            x = self.spk_emb_layer(x)
            x = x.permute(0, 2, 1)  # (B, T, C)
        
        if self.which_decoder == 'transformer':
            x = self.decoder(x, padding_mask=padding_mask_avhubert)     # (B, T, C)
            x = self.decoder_conv(x)
        elif self.which_decoder == 'gru':
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = self.decoder(x, lip_len)    # (B, T, C)
            x = self.decoder_conv(x)
        else:
            x = self.decoder(x)
        return x, None, avhubert_feature
    

class Lip2SP_NAR_Lightweight(nn.Module):
    def __init__(
        self,
        avhubert_config,
        rnn_n_layers,
        rnn_dropout,
        reduction_factor,
        rnn_which_norm,
        out_channels,
        dec_n_layers,
        dec_kernel_size,
        dec_dropout,
        use_spk_emb,
        spk_emb_dim,
    ):
        super().__init__()
        self.resnet = ResEncoder(
            relu_type=avhubert_config.base.resnet_relu_type,
            weights=avhubert_config.base.resnet_weights
        )
        inner_channels = self.resnet.backend_out

        if use_spk_emb:
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        self.encoder = GRUEncoder(
            hidden_channels=inner_channels,
            n_layers=rnn_n_layers,
            dropout=rnn_dropout,
            reduction_factor=reduction_factor,
            which_norm=rnn_which_norm,
        )

        self.decoder = ResTCDecoder(
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
        lip = lip.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        x = self.resnet(lip)    # (B, C, T)
        x = self.encoder(x, lip_len)    # (B, T, C)
        if hasattr(self, 'spk_emb_layer'):
            x = x.permute(0, 2, 1)
            spk_emb = spk_emb.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])
            x = torch.cat([x, spk_emb], dim=1)
            x = self.spk_emb_layer(x)
            x = x.permute(0, 2, 1)
        x = self.decoder(x)
        return x, None, None


class Lip2SP_NAR_AVHUBERT_BOTH(nn.Module):
    def __init__(
            self,
            avhubert_config,
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
            postnet_out_channels,
    ):
        super().__init__()
        self.avhubert_return_res_output = avhubert_return_res_output
        self.which_decoder = which_decoder
        self.reduction_factor = reduction_factor
        self.postnet_out_channels = postnet_out_channels

        self.avhubert = load_avhubert_torch(
            cfg=avhubert_config,
            model_size=avhubert_model_size,
            load_pretrained_weight=load_avhubert_pretrained_weight,
            layer_loaded=avhubert_layer_loaded,
        )
        inner_channels = self.avhubert.encoder_embed_dim

        if use_spk_emb:
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        self.decoder = ResTCDecoder(
            cond_channels=inner_channels,
            out_channels=out_channels,
            inner_channels=inner_channels,
            n_layers=dec_n_layers,
            kernel_size=dec_kernel_size,
            dropout=dec_dropout,
            reduction_factor=reduction_factor,
        )

        self.avhubert_postnet = load_avhubert_torch(
            cfg=avhubert_config,
            model_size=avhubert_model_size,
            load_pretrained_weight=load_avhubert_pretrained_weight,
            layer_loaded=avhubert_layer_loaded,
        )
        self.postnet_out_layer = nn.Linear(self.avhubert_postnet.encoder_embed_dim, postnet_out_channels * reduction_factor)

    def forward(
            self,
            lip,
            audio,
            lip_len,
            feature_len,
            spk_emb,
    ):
        lip = lip.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        padding_mask = torch.arange(lip.shape[2]).unsqueeze(1).to(device=lip.device)
        padding_mask = padding_mask >= lip_len
        padding_mask = padding_mask.permute(1, 0)     # transpose_1
        
        x = self.avhubert(
            video=lip,
            audio=None, 
            return_res_output=self.avhubert_return_res_output,
            padding_mask=padding_mask, 
        )   # (B, T, C)
        
        if hasattr(self, 'spk_emb_layer'):
            x = x.permute(0, 2, 1)  # (B, C, T)
            spk_emb = spk_emb.unsqueeze(-1).expand(spk_emb.shape[0], spk_emb.shape[1], x.shape[-1])
            x = torch.cat([x, spk_emb], dim=1)
            x = self.spk_emb_layer(x)
            x = x.permute(0, 2, 1)  # (B, T, C)

        dec_output = self.decoder(x)     # (B, C, T)
        dec_output = dec_output.permute(0, 2, 1)  # (B, T, C)
        dec_output = dec_output.reshape(dec_output.shape[0], -1, self.reduction_factor, dec_output.shape[2]).reshape(dec_output.shape[0], -1, self.reduction_factor * dec_output.shape[2])
        dec_output = dec_output.permute(0, 2, 1)  # (B, C, T)
        output = self.avhubert_postnet(
            video=None,
            audio=dec_output,
            return_res_output=False,
            padding_mask=padding_mask,
        )   # (B, T, C)
        output = self.postnet_out_layer(output)
        output = output.permute(0, 2, 1)    # (B, C, T)
        output = output.reshape(output.shape[0], self.postnet_out_channels, -1)

        return dec_output, output



import hydra
@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    model = Lip2SP_NAR_AVHUBERT_BOTH(
        avhubert_config=cfg.model.avhubert_config,
        avhubert_model_size=cfg.model.avhubert_model_size,
        avhubert_return_res_output=cfg.model.avhubert_return_res_output,
        load_avhubert_pretrained_weight=cfg.model.load_avhubert_pretrained_weight,
        avhubert_layer_loaded=cfg.model.avhubert_layer_loaded,
        which_encoder=cfg.model.which_encoder,
        rnn_n_layers=cfg.model.rnn_n_layers,
        rnn_dropout=cfg.train.rnn_dropout,
        reduction_factor=cfg.model.reduction_factor,
        rnn_which_norm=cfg.model.rnn_which_norm,
        which_decoder=cfg.model.which_decoder,
        out_channels=cfg.model.avhubert_nfilt,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_kernel_size=cfg.model.tc_kernel_size,
        dec_dropout=cfg.train.dec_dropout,
        use_spk_emb=cfg.train.use_spk_emb,
        spk_emb_dim=cfg.model.spk_emb_dim,
        dec_args=cfg.model.avhubert_dec_args,
        transformer_decoder_num_layers=cfg.model.transformer_decoder_num_layers,
        pos_enc_max_len=int(cfg.model.fps * cfg.model.input_lip_sec),
        postnet_out_channels=cfg.model.n_mel_channels,
    )

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters with requires_grad=True:", num_trainable_params)
    
    lip = torch.rand(2, 1, 88, 88, 250)
    lip_len = torch.randint(100, 250, (lip.shape[0],))
    feature_len = lip_len * cfg.model.reduction_factor
    spk_emb = torch.rand(lip.shape[0], 256)
    dec_output, output = model(
        lip=lip,
        audio=None,
        lip_len=lip_len,
        feature_len=feature_len,
        spk_emb=spk_emb,
    )
    print(dec_output.shape, output.shape)

    

if __name__ == '__main__':
    main()