import torch
from torch import nn

from .net import ResNet3D
from .transformer_remake import Encoder

from .vq_vae import ContentEncoder, VectorQuantizer, ResTCDecoder

class Lip2Sp_VQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.ResNet_GAP = ResNet3D(
            in_channels=3, 
            out_channels=256, 
            inner_channels=128,
            layers=3, 
            dropout=0.5,
            norm_type='in',
        )
        
        self.encoder = Encoder(
            n_layers=1, 
            n_head=4, 
            d_model=256, 
            reduction_factor=2,  
        )

        self.vq = VectorQuantizer(num_embeddings=2048, embedding_dim=256, commitment_cost=0.25)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=256,
            out_channels=80,
            inner_channels=256,
            n_layers=3,
            kernel_size=5,
            dropout=0.5,
            feat_add_channels=80, 
            feat_add_layers=80,
            use_feat_add=False,
            phoneme_classes=53,
            use_phoneme=False,
            n_attn_layer=1,
            n_head=4,
            d_model=256,
            reduction_factor=1,
            use_attention=False,
            compress_rate=2,
            upsample_method='conv'
        )
        
    def forward(self, lip, data_len):
        
        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        enc_output = self.encoder(lip_feature, data_len)  
        
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)
        output = self.decoder(vq, data_len)
    
        all_out = {}
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding

        return all_out