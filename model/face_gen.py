import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        hc = hidden_channels
        in_cs = [in_channels, hc, hc, int(hc * 2), int(hc * 2)]
        out_cs = [hc, hc, int(hc * 2), int(hc * 2), out_channels]
        stride = [2, 1, 2, 1, 1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, stride=s, padding=1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs, out_cs, stride)
        ])
        self.rnn = nn.GRU(out_channels, out_channels, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(int(out_channels * 2), out_channels)

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        out : (B, C, T)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = x.permute(0, 2, 1)  # (B, T, C)
        data_len = torch.clamp(data_len, max=x.shape[1])
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)  
        x = pad_packed_sequence(x, batch_first=True)[0]

        x = self.fc(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        return x


class Generator(nn.Module):
    def __init__(
        self, in_channels, img_hidden_channels, img_cond_channels, 
        feat_channels, feat_cond_channels, mel_enc_hidden_channels, 
        noise_channels, tc_ksize, dropout, is_large):
        super().__init__()
        assert tc_ksize % 2 == 0
        self.noise_channels = noise_channels
        hc = img_hidden_channels
        in_cs = [in_channels, hc, int(hc * 2), int(hc * 4)]
        out_cs = [hc, int(hc * 2), int(hc * 4), int(hc * 8)]

        if tc_ksize == 2:
            padding = 0
        elif tc_ksize == 4:
            padding = 1

        self.audio_enc = MelEncoder(feat_channels, feat_cond_channels, mel_enc_hidden_channels, dropout)
        self.noise_rnn = nn.GRU(noise_channels, noise_channels, num_layers=1, batch_first=True, bidirectional=True)
        self.noise_fc = nn.Linear(noise_channels * 2, noise_channels)

        self.enc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.enc_last_layer = nn.Sequential(
            nn.Conv2d(out_cs[-1], img_cond_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.dec_first_layer = nn.Sequential(
            nn.Conv3d(img_cond_channels + feat_cond_channels + noise_channels, out_cs[-1], kernel_size=3, padding=1),
            nn.BatchNorm3d(out_cs[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if is_large:
            self.dec_expand_layer = nn.Sequential(
                nn.ConvTranspose3d(out_cs[-1], out_cs[-1], kernel_size=(tc_ksize, tc_ksize, 1), stride=(2, 2, 1), padding=(padding, padding, 0)),
                nn.BatchNorm3d(out_cs[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.dec_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(out_c * 2, out_c * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.ConvTranspose3d(out_c * 2, in_c, kernel_size=(tc_ksize, tc_ksize, 1), stride=(2, 2, 1), padding=(padding, padding, 0)),
                nn.BatchNorm3d(in_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for in_c, out_c, in zip(list(reversed(in_cs))[:-1], list(reversed(out_cs))[:-1])
        ])
        self.dec_last_layer = nn.Sequential(
            nn.Conv3d(out_cs[0] * 2, out_cs[0] * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ConvTranspose3d(out_cs[0] * 2, in_cs[0], kernel_size=(tc_ksize, tc_ksize, 1), stride=(2, 2, 1), padding=(padding, padding, 0)),
        )

    def forward(self, lip, feature, data_len):
        """
        lip : (B, C, H, W)
        feature : (B, C, T)
        out : (B, C, H, W, T)
        """
        B, C, H, W = lip.shape
        enc_out = lip
        fmaps = []
        for layer in self.enc_layers:
            enc_out = layer(enc_out)
            fmaps.append(enc_out)

        # 画像から得られる見た目についての特徴表現
        lip_rep = self.enc_last_layer(enc_out)  # (B, C, H, W)
        lip_rep = torch.mean(lip_rep, dim=(2, 3), keepdim=True)     # (B, C, 1, 1)

        # 音響特徴量から得られる特徴表現
        feat_rep = self.audio_enc(feature, data_len)  # (B, C, T)

        # ノイズを生成
        noise = torch.normal(
            mean=0, std=0.6, size=(feat_rep.shape[0], feat_rep.shape[-1], self.noise_channels)
        ).to(device=lip.device, dtype=lip.dtype)    # (B, T, C)

        data_len = torch.clamp(data_len, max=noise.shape[1])
        noise = pack_padded_sequence(noise, data_len.cpu(), batch_first=True, enforce_sorted=False)
        noise_rep, _ = self.noise_rnn(noise)    
        noise_rep = pad_packed_sequence(noise_rep, batch_first=True)[0]

        noise_rep = self.noise_fc(noise_rep)
        noise_rep = noise_rep.permute(0, 2, 1)  # (B, C, T)

        lip_rep = lip_rep.unsqueeze(-1)     # (B, C, 1, 1, 1)
        lip_rep = lip_rep.expand(-1, -1, -1, -1, feat_rep.shape[-1])  # (B, C, 1, 1, T)
        feat_rep = feat_rep.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        noise_rep = noise_rep.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        out = torch.cat([lip_rep, feat_rep, noise_rep], dim=1)

        # 音声と画像から発話内容に対応した動画を合成
        out = F.interpolate(out, size=(3, 3, 75))
        out = self.dec_first_layer(out)     # (B, C, H, W, T)
        if hasattr(self, "dec_expand_layer"):
            out = self.dec_expand_layer(out)
            
        for layer, fmap in zip(self.dec_layers, reversed(fmaps)):
            fmap = fmap.unsqueeze(-1).expand(-1, -1, -1, -1, out.shape[-1])
            out = torch.cat([out, fmap], dim=1)
            out = layer(out)
        
        out = torch.cat([out, fmaps[0].unsqueeze(-1).expand(-1, -1, -1, -1, out.shape[-1])], dim=1)
        out = self.dec_last_layer(out)
        return out


class FrameDiscriminator(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        in_cs = [in_channels, 32, 64, 128]
        out_cs = [32, 64, 128, 256]

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.last_layer = nn.Linear(out_cs[-1], 1)

    def forward(self, lip, lip_cond):
        """
        ランダムに選択した1フレームの画像に対する判別を行う
        lip : (B, C, H, W)      ランダムに選択したフレーム
        lip_cond : (B, C, H, W)     合成に使用したフレーム
        out : (B, 1)
        """
        lip = torch.cat([lip, lip_cond], dim=1)
        out = lip
        for layer in self.layers:
            out = layer(out)
        out = torch.mean(out, dim=(2, 3))
        out = self.last_layer(out)
        return out


class MultipleFrameDiscriminator(nn.Module):
    def __init__(self, in_channels, dropout, analysis_len):
        super().__init__()
        in_cs = [in_channels, 32, 64, 128]
        out_cs = [32, 64, 128, 256]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, stride=(2, 2, 1), padding=1),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.last_layer = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(out_cs[-1] * analysis_len, 1),
        )

    def forward(self, lip):
        """
        lip : (B, C, H, W, T)
        """
        out = lip
        for layer in self.layers:
            out = layer(out)
        out = torch.mean(out, dim=(2, 3))   # (B, C, T)
        out = self.last_layer(out)  # (B, 1)
        return out


class SequenceDiscriminator(nn.Module):
    def __init__(self, in_channels, feat_channels, dropout, analysis_len):
        super().__init__()
        in_cs_lip = [in_channels, 32, 64, 128]
        out_cs_lip = [32, 64, 128, 256]
        self.lip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1)),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs_lip, out_cs_lip)
        ])

        in_cs_feat = [feat_channels, 128, 128, 256, 256]
        out_cs_feat = [128, 128, 256, 256, out_cs_lip[-1]]
        stride = [2, 1, 2, 1, 1]
        self.feat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs_feat, out_cs_feat, stride)
        ])
        self.lip_rnn = nn.GRU(out_cs_lip[-1], out_cs_lip[-1] // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.feat_rnn = nn.GRU(out_cs_lip[-1], out_cs_lip[-1] // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.out_layers= nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(out_cs_lip[-1] * analysis_len), 1)
        )
    
    def forward(self, lip, feature):
        """
        動画と音響特徴量を利用し,系列全体から判定する
        lip : (B, C, H, W, T)
        feature : (B, T, C)
        """
        # print(f"seq_disc")
        # print(f"lip = {lip.shape}, feature = {feature.shape}")
        lip_rep = lip
        for layer in self.lip_layers:
            lip_rep = layer(lip_rep)
        lip_rep = torch.mean(lip_rep, dim=(2, 3))   # (B, C, T)
        
        feat_rep = feature
        for layer in self.feat_layers:
            feat_rep = layer(feat_rep)
        
        lip_rep = lip_rep.permute(0, 2, 1)  # (B, T, C)
        feat_rep = feat_rep.permute(0, 2, 1)    #(B, T, C)

        # padding部分の考慮はせず、discriminatorにノイズを与える
        lip_rep, _ = self.lip_rnn(lip_rep)
        feat_rep, _ = self.feat_rnn(feat_rep)
        out = lip_rep + feat_rep
        out = self.out_layers(out)      # (B, 1)
        return out


class SyncDiscriminator(nn.Module):
    def __init__(self, in_channels, feat_channels,  dropout):
        super().__init__()
        in_cs_lip = [in_channels, 32, 64, 128]
        out_cs_lip = [32, 64, 128, 256]
        self.lip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs_lip, out_cs_lip)
        ])

        in_cs_feat = [feat_channels, 128, 128, 256]
        out_cs_feat = [128, 128, 256, 256]
        stride = [2, 1, 2, 1]
        self.feat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs_feat, out_cs_feat, stride)
        ])
        self.out_fc = nn.Linear(256, 1)

    def forward(self, lip, feature):
        """
        適当な長さの音声と口唇動画のペアから,それらが同期しているかどうか判別
        lip : (B, C, H, W, T)
        feature : (B, C, T)
        """
        # print(f"sync_disc")
        # print(f"lip = {lip.shape}, feature = {feature.shape}")
        lip_rep = lip
        for layer in self.lip_layers:
            lip_rep = layer(lip_rep)
        lip_rep = torch.mean(lip_rep, dim=(2, 3))   # (B, C, T)

        feat_rep = feature
        for layer in self.feat_layers:
            feat_rep = layer(feat_rep)      # (B, C, T)

        out = lip_rep + feat_rep
        out = torch.mean(out, dim=-1)   # (B, C)
        out = self.out_fc(out)
        return out