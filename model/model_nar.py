import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder
    from model.conformer.encoder import ConformerEncoder
    from model.audio_enc import SpeakerEncoder, ContentEncoder
    from model.nar_decoder import TCDecoder, GatedTCDecoder, ResTCDecoder
    from model.vq import VQ
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder
    from .conformer.encoder import ConformerEncoder
    from .audio_enc import SpeakerEncoder, ContentEncoder
    from .nar_decoder import TCDecoder, GatedTCDecoder, ResTCDecoder
    from .vq import VQ


class Lip2SP_NAR(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels, norm_type,
        d_model, n_layers, n_head, conformer_conv_kernel_size,
        dec_n_layers, dec_inner_channels, dec_kernel_size,
        feat_add_channels, feat_add_layers, 
        n_speaker, spk_emb_dim,
        which_encoder, which_decoder, apply_first_bn, use_feat_add, phoneme_classes, use_phoneme, 
        upsample_method, compress_rate,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False):
        super().__init__()

        assert d_model % n_head == 0
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.use_gc = use_gc

        self.first_batch_norm = nn.BatchNorm3d(in_channels)

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=d_model, 
            inner_channels=res_inner_channels,
            layers=res_layers, 
            dropout=res_dropout,
            norm_type=norm_type,
        )

        # encoder
        if self.which_encoder == "transformer":
            self.encoder = Encoder(
                n_layers=n_layers, 
                n_head=n_head, 
                d_model=d_model, 
                reduction_factor=reduction_factor,  
            )
        elif self.which_encoder == "conformer":
            self.encoder = ConformerEncoder(
                encoder_dim=d_model, 
                num_layers=n_layers, 
                num_attention_heads=n_head, 
                conv_kernel_size=conformer_conv_kernel_size,
                reduction_factor=reduction_factor,
            )

        self.emb_layer = nn.Embedding(n_speaker, spk_emb_dim)

        # decoder
        if self.which_decoder == "simple_tc":
            self.decoder = TCDecoder(
                cond_channels=d_model,
                out_channels=out_channels,
                inner_channels=dec_inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
            )
        elif self.which_decoder == "gated_tc":
            self.decoder = GatedTCDecoder(
                cond_channels=d_model,
                out_channels=out_channels,
                inner_channels=dec_inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
            )
        elif self.which_decoder == "res_tc":
            self.decoder = ResTCDecoder(
                cond_channels=d_model,
                out_channels=out_channels,
                inner_channels=dec_inner_channels,
                n_layers=dec_n_layers,
                kernel_size=dec_kernel_size,
                dropout=dec_dropout,
                feat_add_channels=feat_add_channels, 
                feat_add_layers=feat_add_layers,
                use_feat_add=use_feat_add,
                phoneme_classes=phoneme_classes,
                use_phoneme=use_phoneme,
                spk_emb_dim=spk_emb_dim,
                upsample_method=upsample_method,
                compress_rate=compress_rate,
            )

    def forward(self, lip=None, feature=None, data_len=None, gc=None):
        output = feat_add_out = phoneme = None

        # resnet
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)

        # speaker embedding
        if gc is not None:
            spk_emb = self.emb_layer(gc)
        else:
            spk_emb = None

        # decoder
        output, feat_add_out, phoneme = self.decoder(enc_output, spk_emb=spk_emb)
        
        return output, feat_add_out, phoneme


if __name__ == "__main__":
    emb_layer = nn.Embedding(2, 10)
    speaker = torch.tensor([1, 0, 1, 1, 0, 1]).to(torch.long)   # speakerは(B,)にしとく (B,) -> (B, C)
    spk_emb = emb_layer(speaker)
    breakpoint()