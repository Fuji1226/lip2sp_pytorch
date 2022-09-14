import sys
from pathlib import Path
sys.path.append(Path("~/lip2sp_pytorch").expanduser())
sys.path.append(Path("~/lip2sp_pytorch/model").expanduser())

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.net import ResNet3D
    from model.transformer_remake import Encoder
    from model.conformer.encoder import ConformerEncoder
    from model.audio_enc import SpeakerEncoder, ContentEncoder
    from model.nar_decoder import TCDecoder, GatedTCDecoder, ResTCDecoder
    from model.vae import VAE
except:
    from .net import ResNet3D
    from .transformer_remake import Encoder
    from .conformer.encoder import ConformerEncoder
    from .audio_enc import SpeakerEncoder, ContentEncoder
    from .nar_decoder import TCDecoder, GatedTCDecoder, ResTCDecoder
    from .vae import VAE


class Lip2SP_VAE(nn.Module):
    def __init__(
        self, in_channels, out_channels, res_layers, res_inner_channels,
        d_model, n_layers, n_head, conformer_conv_kernel_size,
        dec_n_layers, dec_inner_channels, dec_kernel_size,
        feat_add_channels, feat_add_layers, 
        vae_emb_dim, spk_emb_dim,
        which_encoder, which_decoder, apply_first_bn, use_feat_add, phoneme_classes, use_phoneme,
        upsample_method,  compress_rate,
        dec_dropout, res_dropout, reduction_factor=2, use_gc=False):
        super().__init__()
        assert use_phoneme == False
        assert d_model % n_head == 0
        assert compress_rate == 2 or compress_rate == 4
        self.which_encoder = which_encoder
        self.which_decoder = which_decoder
        self.apply_first_bn = apply_first_bn
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.use_gc = use_gc
        self.compress_rate = compress_rate

        self.first_batch_norm = nn.BatchNorm3d(in_channels)

        self.ResNet_GAP = ResNet3D(
            in_channels=in_channels, 
            out_channels=d_model, 
            inner_channels=res_inner_channels,
            layers=res_layers, 
            dropout=res_dropout,
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

        # audio encoder
        self.speaker_enc = SpeakerEncoder(
            in_channels=out_channels,
            out_channels=spk_emb_dim,
        )
        self.content_enc = ContentEncoder(
            in_channels=out_channels,
            out_channels=d_model,
            n_attn_layer=n_layers,
            n_head=n_head,
            reduction_factor=reduction_factor,
        )

        self.compress_layer_audio = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.compress_layer_lip = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # vae
        self.vae_lip = VAE(d_model, vae_emb_dim)
        self.vae_audio = VAE(d_model, vae_emb_dim)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=vae_emb_dim,
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

    def forward_(self, lip=None, feature=None, feat_add=None, feature_ref=None, data_len=None, gc=None):
        """
        口唇動画を入力する際にはResnetとencoder以外のパラメータは更新しない(事前学習済み)
        """
        output = feat_add_out = phoneme = spk_emb = mu = logvar = z = enc_output = None
        
        # 口唇動画を入力した場合
        if lip is not None:
            assert feature is not None

            # resnet
            if self.apply_first_bn:
                lip = self.first_batch_norm(lip)
            lip_feature = self.ResNet_GAP(lip)
            
            # encoder
            enc_output = self.encoder(lip_feature, data_len)    # (B, T, C)
            if self.compress_rate == 4:
                enc_outout = self.compress_layer_lip(enc_output.permute(0, 2, 1)).permute(0, 2, 1)
            
            # vae
            mu, logvar, z = self.vae_lip(enc_output)    # (B, T, C)

            # speaker embedding
            with torch.no_grad():
                if feature_ref is None:
                    spk_emb = self.speaker_enc(feature)     # (B, C)
                else:
                    spk_emb = self.speaker_enc(feature_ref)     # (B, C)
                
        # 音響特徴量のみを入力した場合
        elif feature is not None:
            assert lip is None

            # speaker embedding
            if feature_ref is None:
                spk_emb = self.speaker_enc(feature)     # (B, C)
            else:
                spk_emb = self.speaker_enc(feature_ref)     # (B, C)

            # content encoder
            enc_output = self.content_enc(feature)     # (B, T, C)

            # vae
            mu, logvar, z = self.vae_audio(enc_output)    # (B, T, C)
        
        # decoder
        if lip is not None:
            with torch.no_grad():
                output, feat_add_out, phoneme = self.decoder(z, spk_emb, feat_add)
        else:
            output, feat_add_out, phoneme = self.decoder(z, spk_emb, feat_add)

        return output, feat_add_out, phoneme, spk_emb, mu, logvar, z, enc_output

    def forward(self, lip, feature=None, feat_add=None, feature_ref=None, data_len=None):
        output_feat = feat_add_out_feat = phoneme_feat = output_lip = feat_add_out_lip = phoneme_lip = mu_lip = logvar_lip = z_lip = mu_feat = logvar_feat = z_feat = spk_emb = None

        # resnet
        if self.apply_first_bn:
            lip = self.first_batch_norm(lip)
        lip_feature = self.ResNet_GAP(lip)
        
        # encoder
        enc_output_lip = self.encoder(lip_feature, data_len)    # (B, T, C)
        if self.compress_rate == 4:
            enc_outout_lip = self.compress_layer_lip(enc_output_lip.permute(0, 2, 1)).permute(0, 2, 1)
        
        # vae(lip)
        mu_lip, logvar_lip, z_lip = self.vae_lip(enc_output_lip)    # (B, T, C)

        # content encoder
        enc_output_feat = self.content_enc(feature)     # (B, T, C)

        # vae(feature)
        mu_feat, logvar_feat, z_feat = self.vae_audio(enc_output_feat)    # (B, T, C)

        # speaker embedding
        if feature_ref is None:
            spk_emb = self.speaker_enc(feature)     # (B, C)
        else:
            spk_emb = self.speaker_enc(feature_ref)     # (B, C)

        # decoder
        output_feat, feat_add_out_feat, phoneme_feat = self.decoder(z_feat, spk_emb, feat_add)

        with torch.no_grad():
            output_lip, feat_add_out_lip, phoneme_lip = self.decoder(z_lip, spk_emb, feat_add)
        
        return output_feat, feat_add_out_feat, phoneme_feat, output_lip, feat_add_out_lip, phoneme_lip, mu_lip, logvar_lip, z_lip, mu_feat, logvar_feat, z_feat, spk_emb
