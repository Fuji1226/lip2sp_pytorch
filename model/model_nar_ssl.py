import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import torch
import torch.nn as nn
from utils import (
    load_avhubert,
    load_raven,
    load_vatlm,
)


class Lip2SpeechSSL(nn.Module):
    def __init__(
            self,
            cfg,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.model.use_ssl_model == 'avhubert':
            self.avhubert = load_avhubert(cfg.model.avhubert_config)
            hidden_channels = self.avhubert.encoder_embed_dim
        elif cfg.model.use_ssl_model == 'raven':
            self.raven = load_raven(cfg.model.raven_config)
            hidden_channels = self.raven.attention_dim
        elif cfg.model.use_ssl_model == 'vatlm':
            self.vatlm = load_vatlm(cfg.model.vatlm_config)
            hidden_channels = self.vatlm.encoder_embed_dim
        elif cfg.model.use_ssl_model == 'ensemble':
            self.avhubert = load_avhubert(cfg.model.avhubert_config)
            self.raven = load_raven(cfg.model.raven_config)
            self.vatlm = load_vatlm(cfg.model.vatlm_config)
            hidden_channels = self.avhubert.encoder_embed_dim
            self.fuse_layer = nn.Linear(
                self.avhubert.encoder_embed_dim + self.raven.attention_dim + self.vatlm.encoder_embed_dim,
                hidden_channels,
            )
        
        if cfg.train.use_spk_emb:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg.model.spk_emb_dim,
                hidden_channels,
            )
        
    def extract_feature_avhubert(
            self,
            lip,
            lip_len,
            audio,
    ):
        padding_mask = torch.arange(lip.shape[2]).unsqueeze(0).expand(lip.shape[0], -1).to(device=lip.device)
        padding_mask = padding_mask > lip_len.unsqueeze(-1)
        x = self.avhubert(
            video=lip,
            audio=audio, 
            return_res_output=False,
            padding_mask=padding_mask, 
        )   # (B, T, C)
        return x
    
    def extract_feature_raven(
            self,
            lip,
            lip_len,
            audio,
    ):
        padding_mask = torch.arange(lip.shape[2]).unsqueeze(0).expand(lip.shape[0], -1).to(device=lip.device)
        padding_mask = padding_mask <= lip_len.unsqueeze(-1)    # True for unmasked positions
        padding_mask = padding_mask.unsqueeze(1)    # (B, 1, T)
        x, _ = self.raven(
            xs=lip,
            masks=padding_mask,
        )   # (B, T, C)
        return x
    
    def extract_feature_vatlm(
            self,
            lip,
            lip_len,
            audio,
    ):
        padding_mask = torch.arange(lip.shape[2]).unsqueeze(0).expand(lip.shape[0], -1).to(device=lip.device)
        padding_mask = padding_mask > lip_len.unsqueeze(-1)
        x = self.vatlm(
            video=lip,
            audio=audio,
            padding_mask=padding_mask,
        )   # (B, T, C)
        return x

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

        if self.cfg.model.use_ssl_model == 'avhubert':
            feature = self.extract_feature_avhubert(lip, lip_len, audio)
        elif self.cfg.model.use_ssl_model == 'raven':
            feature = self.extract_feature_raven(lip, lip_len, audio)
        elif self.cfg.model.use_ssl_model == 'vatlm':
            feature = self.extract_feature_vatlm(lip, lip_len, audio)
        elif self.cfg.model.use_ssl_model == 'ensemble':
            feature_avhubert = self.extract_feature_avhubert(lip, lip_len, audio)
            feature_raven = self.extract_feature_raven(lip, lip_len, audio)
            feature_vatlm = self.extract_feature_vatlm(lip, lip_len, audio)
            feature = torch.concat([feature_avhubert, feature_raven, feature_vatlm], dim=-1)
            feature = self.fuse_layer(feature)

        if self.cfg.train.use_spk_emb:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)   # (B, T, C)
            feature = torch.cat([feature, spk_emb], dim=-1)
            feature = self.spk_emb_layer(feature)

        print(feature.shape)
        return


import hydra
@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    breakpoint()

    lip = torch.rand(2, 1, 88, 88, 250)
    lip_len = torch.randint(100, 250, (lip.shape[0],))
    feature = torch.rand(lip.shape[0], 80, 1000)
    feature_len = lip_len * 4
    spk_emb = torch.rand(lip.shape[0], 256)
    model = Lip2SpeechSSL(cfg)

    lip = lip.to(device)
    lip_len = lip_len.to(device)
    feature = feature.to(device)
    feature_len = feature_len.to(device)
    spk_emb = spk_emb.to(device)
    model = model.to(device)

    model.train()
    model(
        lip=lip,
        audio=None,
        lip_len=lip_len,
        spk_emb=spk_emb,
    )


if __name__ == '__main__':
    main()