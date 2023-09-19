import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from nar_decoder import ResTCDecoder, LinearDecoder
from rnn import GRUEncoder
from avhubert import Config, MyAVHubertModel, TransformerEncoder


def load_avhubert_torch(ckpt_path, model_size, load_pretrained_weight, layer_loaded):
    cfg = Config(model_size)
    avhubert = MyAVHubertModel(cfg)

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
        transformer_args,
        out_channels,
        reduction_factor,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.transformer = TransformerEncoder(transformer_args)
        self.fc = nn.Linear(self.transformer.embedding_dim, out_channels * reduction_factor)

    def forward(self, x, padding_mask, layer=None):
        '''
        x : (B, T, C)
        padding_mask : (B, T)
        '''
        x, _ = self.transformer(x, padding_mask=padding_mask, layer=layer)
        x = self.fc(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = x.reshape(x.shape[0], self.out_channels, -1)
        return x


class Lip2SP_NAR_AVHubert(nn.Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        self.avhubert_return_res_output = avhubert_return_res_output
        self.avhubert = load_avhubert_torch(
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
        elif which_decoder == 'avhubert_transformer':
            self.decoder = TransformerDecoder(
                transformer_args=dec_args,
                out_channels=out_channels,
                reduction_factor=reduction_factor,
            )

    def forward(
        self,
        lip,
        lip_len,
        spk_emb,
    ):
        '''
        lip : (B, C, H, W, T)
        lip_len : (B,)
        spk_emb : (B, C)
        '''
        lip = lip.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
        padding_mask_avhubert = torch.zeros(lip.shape[0], lip.shape[2]).to(device=lip.device, dtype=torch.bool)     # (B, T)
        for i, l in enumerate(lip_len):
            padding_mask_avhubert[i, l:] = True

        x = self.avhubert(
            lip, 
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

        x = self.decoder(x, padding_mask=padding_mask_avhubert, layer=None)
        return x, None, None
    


"""Input shape: Time x Batch x Channel

Args:
    key_padding_mask (ByteTensor, optional): mask to exclude
        keys that are pads, of shape `(batch, src_len)`, where
        padding elements are indicated by 1s.
    need_weights (bool, optional): return the attention weights,
        averaged over heads (default: False).
    attn_mask (ByteTensor, optional): typically used to
        implement causal attention, where the mask prevents the
        attention from looking forward in time (default: None).
    before_softmax (bool, optional): return the raw attention
        weights and values before the attention softmax.
    need_head_weights (bool, optional): return the attention
        weights for each head. Implies *need_weights*. Default:
        return the average attention weights over all heads.
# """
# if __name__ == '__main__':
#     ckpt_path = Path('~/av_hubert_data/base_vox_iter5_torch.ckpt').expanduser()
#     model_size = 'base'
#     avhubert = load_avhubert_torch(ckpt_path, model_size)
#     B = 4
#     T = 250
#     C = 1
#     features = torch.rand(B, T, C)
#     padding_mask = torch.zeros(B, T)
#     data_len_list = [50, 100, 100, 300]
#     for i, data_len in enumerate(data_len_list):
#         padding_mask[i, data_len:] = 1
#     padding_mask = avhubert.forward_padding_mask(features, padding_mask)