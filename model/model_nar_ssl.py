import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import (
    load_avhubert,
    load_raven,
    load_vatlm,
)
from model.avhubert import ResEncoder


class ResBlock(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = x
        out = self.conv_layers(x)
        return out + res


class ResConvDecoder(nn.Module):
    def __init__(
            self,
            cfg,
            hidden_channels,
    ):
        super().__init__()
        self.cfg = cfg
        self.conv_layers = []
        for i in range(cfg.model.decoder.n_conv_layers):
            self.conv_layers.append(
                ResBlock(
                    hidden_channels=hidden_channels,
                    kernel_size=cfg.model.decoder.conv_kernel_size,
                    dropout=cfg.model.decoder.dropout,
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.out_layer = nn.Conv1d(hidden_channels, cfg.model.n_mel_channels * cfg.model.reduction_factor, kernel_size=1)

    def forward(
            self,
            x,
    ):
        '''
        x: (B, T, C)
        '''
        x = x.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.out_layer(x)
        x = x.permute(0, 2, 1)      # (B, T, C)
        x = x.reshape(x.shape[0], -1, self.cfg.model.n_mel_channels)
        x = x.permute(0, 2, 1)      # (B, C, T)
        return x


class Lip2SpeechSSL(nn.Module):
    def __init__(
            self,
            cfg,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.model.model_name == 'avhubert':
            self.avhubert = load_avhubert(cfg.model.avhubert_config)
            hidden_channels = self.avhubert.encoder_embed_dim
        elif cfg.model.model_name == 'raven':
            self.raven = load_raven(cfg.model.raven_config)
            hidden_channels = self.raven.attention_dim
        elif cfg.model.model_name == 'vatlm':
            self.vatlm = load_vatlm(cfg.model.vatlm_config)
            hidden_channels = self.vatlm.encoder_embed_dim
        elif cfg.model.model_name == 'ensemble':
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

        self.decoder = ResConvDecoder(cfg, hidden_channels)
        
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

        if self.cfg.model.model_name == 'avhubert':
            feature = self.extract_feature_avhubert(lip, lip_len, audio)
        elif self.cfg.model.model_name == 'raven':
            feature = self.extract_feature_raven(lip, lip_len, audio)
        elif self.cfg.model.model_name == 'vatlm':
            feature = self.extract_feature_vatlm(lip, lip_len, audio)
        elif self.cfg.model.model_name == 'ensemble':
            feature_avhubert = self.extract_feature_avhubert(lip, lip_len, audio)
            feature_raven = self.extract_feature_raven(lip, lip_len, audio)
            feature_vatlm = self.extract_feature_vatlm(lip, lip_len, audio)
            feature = torch.concat([feature_avhubert, feature_raven, feature_vatlm], dim=-1)
            feature = self.fuse_layer(feature)

        if self.cfg.train.use_spk_emb:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, feature.shape[1], -1)   # (B, T, C)
            feature = torch.cat([feature, spk_emb], dim=-1)
            feature = self.spk_emb_layer(feature)

        output = self.decoder(feature)

        return output


class Lip2SpeechLightWeight(nn.Module):
    def __init__(
            self,
            cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.resnet = ResEncoder(
            relu_type=cfg.model.avhubert_config.base.resnet_relu_type,
            weights=cfg.model.avhubert_config.base.resnet_weights,
            cfg=cfg.model.avhubert_config.base,
        )
        hidden_channels = self.resnet.backend_out
        if cfg.train.use_spk_emb:
            self.spk_emb_layer = nn.Linear(
                hidden_channels + cfg.model.spk_emb_dim,
                hidden_channels,
            )
        
        self.dropout = nn.Dropout(cfg.model.lightweight.dropout)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=cfg.model.lightweight.lstm_n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        self.decoder = ResConvDecoder(cfg, hidden_channels)

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

        x = self.dropout(x).permute(0, 2, 1)    # (B, T, C)
        B, T, C = x.shape
        lip_len = torch.clamp(lip_len, max=T)
        x = pack_padded_sequence(x, lip_len.cpu(), batch_first=True, enforce_sorted=False)
        x, (hn, cn) = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]
        if x.shape[1] < T:
            zero_pad = torch.zeros(x.shape[0], T - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)
        x = self.fc(x)  # (B, T, C)

        if self.cfg.train.use_spk_emb:
            spk_emb = spk_emb.unsqueeze(1).expand(-1, x.shape[1], -1)   # (B, T, C)
            x = torch.cat([x, spk_emb], dim=-1)
            x = self.spk_emb_layer(x)

        x = self.decoder(x)
        return x


# import hydra
# @hydra.main(version_base=None, config_name="config", config_path="../conf")
# def main(cfg):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)

#     lip = torch.rand(2, 1, 88, 88, 125)
#     lip_len = torch.randint(lip.shape[-1] // 2, lip.shape[-1], (lip.shape[0],))
#     feature = torch.rand(lip.shape[0], 80, 1000)
#     feature_len = lip_len * 4
#     spk_emb = torch.rand(lip.shape[0], 256)
#     model = Lip2SpeechSSL(cfg)
#     # model = Lip2SpeechLightWeight(cfg)

#     lip = lip.to(device)
#     lip_len = lip_len.to(device)
#     feature = feature.to(device)
#     feature_len = feature_len.to(device)
#     spk_emb = spk_emb.to(device)
#     model = model.to(device)

#     ckpt_path_avhubert = Path('/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:11:25_19-16-55/2.ckpt')
#     ckpt_path_raven = Path('/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:11:25_19-18-17/2.ckpt')
#     ckpt_path_vatlm = Path('/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:11:25_19-19-35/2.ckpt')
#     ckpt_avhubert = torch.load(str(ckpt_path_avhubert), map_location=device)['model']
#     ckpt_avhubert = {name: param for name, param in ckpt_avhubert.items() if 'avhubert.' in name}
#     ckpt_raven = torch.load(str(ckpt_path_raven), map_location=device)['model']
#     ckpt_raven = {name: param for name, param in ckpt_raven.items() if 'raven.' in name}
#     ckpt_vatlm = torch.load(str(ckpt_path_vatlm), map_location=device)['model']
#     ckpt_vatlm = {name: param for name, param in ckpt_vatlm.items() if 'vatlm.' in name}

#     model_dict = model.state_dict()
#     match_dict = {name: param for name, param in ckpt_avhubert.items() if name in model_dict}
#     model.load_state_dict(match_dict, strict=False)

#     model_dict = model.state_dict()
#     match_dict = {name: param for name, param in ckpt_raven.items() if name in model_dict}
#     model.load_state_dict(match_dict, strict=False)

#     model_dict = model.state_dict()
#     match_dict = {name: param for name, param in ckpt_vatlm.items() if name in model_dict}
#     model.load_state_dict(match_dict, strict=False)

#     for name, param in model.named_parameters():
#         if name in ckpt_avhubert and not torch.equal(param, ckpt_avhubert[name]):
#             print(name)
#         if name in ckpt_raven and not torch.equal(param, ckpt_raven[name]):
#             print(name)            
#         if name in ckpt_vatlm and not torch.equal(param, ckpt_vatlm[name]):
#             print(name)

#     cnt1 = 0
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             cnt1 += param.numel()

#     for name, param in model.named_parameters():
#         if 'avhubert.' in name or 'raven.' in name or 'vatlm.' in name:
#             param.requires_grad = False
    
#     cnt2 = 0
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             cnt2 += param.numel()
#             print(name)

#     print(cnt1, cnt2)

    
#     # model.train()
#     # output = model(
#     #     lip=lip,
#     #     audio=None,
#     #     lip_len=lip_len,
#     #     spk_emb=spk_emb,
#     # )
#     # print(output.shape)


# if __name__ == '__main__':
#     main()