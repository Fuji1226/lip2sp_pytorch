import torch 
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


class JCUDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, emb_in=10, n_features=16):
        super().__init__()
        # 共有する層
        self.share = [
            nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, stride=2),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(128, 512, kernel_size=5, stride=2),
                nn.LeakyReLU(0.2),
            ),
        ]
        self.share = nn.ModuleList(self.share)

        # unconditional layers
        self.uncond = [
            nn.Sequential(
                nn.Conv1d(512, 128, kernel_size=5, stride=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(128, out_channels, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2),
            ),
        ]
        self.uncond = nn.ModuleList(self.uncond)

        # conditional layers
        self.embed = nn.Embedding(emb_in, n_features)
        self.fc = nn.Linear(n_features, out_features=512)
        self.cond = [
            nn.Sequential(
                nn.Conv1d(512, 128, kernel_size=5, stride=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(128, out_channels, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2),
            ),
        ]
        self.cond = nn.ModuleList(self.cond)

    def forward(self, x, s):
        """
        x : (B, mel_channels, t)
        s : (B, 1)
        """
        fmaps_share = []
        fmaps_uncond = []
        fmaps_cond = []

        # share
        for idx, layer in enumerate(self.share):
            if idx == 0:
                out_share = layer(x)
            else:
                out_share = layer(out_share)
            fmaps_share.append(out_share)

        out_uncond = out_share
        out_cond = out_share
        
        # unconditional output
        for idx, layer in enumerate(self.uncond):
            out_uncond = layer(out_uncond)
            fmaps_uncond.append(out_uncond)

        # conditional output
        s = self.embed(s)   # (B, 1) -> (B, 1, n_features)
        condition = self.fc(s)      # (B, 1, n_features) -> (B, 1, 512)
        condition = torch.transpose(condition, 1, 2)    # (B, 1, 512) -> (B, 512, 1)
        # 論文ではconditionを時間方向に拡大すると書いてあったけど、expandを使うということなのかよくわからない
        # 一応VocGANではconditionであるメルスペクトログラムを時間方向に結合していた
        # この実装では拡大せずに結合している
        out_cond = torch.cat([out_cond, condition], dim=2)      # out_condとconditionを時間方向に結合

        for idx, layer in enumerate(self.uncond):
            out_cond = layer(out_cond)
            fmaps_cond.append(out_cond)

        return out_share, fmaps_share, out_uncond, fmaps_uncond, s, condition, out_cond, fmaps_cond


class UNetDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        """
        原論文が単話者なので、一旦単話者を想定して実装
        encoderの特徴量の結合の仕方が論文に詳しく書いていないので微妙。とりあえずUNetと同様に、チャンネル方向に結合している。
        複数話者にも適応したい
        また、ResUNetのアイデアも使ってみたい
        """
        self.encoder = [
            nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),
        ]
        self.encoder = nn.ModuleList(self.encoder)
        self.encoder_out = nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1)

        self.decoder = [
            nn.Sequential(
                weight_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                weight_norm(nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
            ),
        ]

    def forward(self, x):
        """
        x : (B, 1, mel_channels, t)
        """
        fmaps_enc = []

        # encoder
        for idx, layer in enumerate(self.encoder):
            if idx == 0:
                out_enc = layer(x)
                fmaps_enc.append(out_enc)
            else:
                out_enc = layer(out_enc)
                fmaps_enc.append(out_enc)
        out_enc = self.encoder_out(out_enc)

        # decoder
        for idx, layer in enumerate(self.decoder):
            """
            decoder1層目 : fmaps_enc[-1]を入力
            decoder2層目 : decoder1層目の出力とfmaps_enc[-2]を結合したものを入力
            decoder3層目 : decoder2層目の出力とfmaps_enc[-3]を結合したものを入力
            decoder4層目 : decoder3層目の出力とfmaps_enc[-4]を結合したものを入力
            decoder5層目 : decoder4層目の出力とfmaps_enc[-5]を結合したものを入力
            decoder6層目 : decoder5層目の出力とfmaps_enc[-6]を結合したものを入力
            """
            if idx == 0:
                out_dec = layer(fmaps_enc[-1])
            else:
                out_dec = torch.cat([out_dec, fmaps_enc[-1-idx]], dim=1)    # とりあえずUNetと同様にチャンネル方向に結合している。原論文に詳しく書いていないので一緒かはわからない。
                out_dec = layer(out_dec)

        return out_enc, fmaps_enc, out_dec


def main():
    B = 10
    mel_channels = 80
    t = 1000
    global_feature = torch.arange(B).reshape(B, 1)      # 複数話者の場合を想定
    x = torch.rand(B, mel_channels, t)

    """
    # GANSpeechの場合
    GANSpeech = JCUDiscriminator(in_channels=mel_channels, out_channels=1, emb_in=B)
    out_share, fmaps_share, out_uncond, fmaps_uncond, s, condition, out_cond, fmaps_cond = GANSpeech(x, global_feature)
    print(f"out_share.shape = {out_share.shape}")
    print(f"out_uncond.shape = {out_uncond.shape}")
    print(f"s.shape = {s.shape}")
    print(f"condition.shape = {condition.shape}")
    print(f"out_cond.shape = {out_cond.shape}")
    """

    # U-Netの場合
    x_add = x.unsqueeze(1)  # (B, mel_channels, t) -> (B, 1, mel_channels, t)
    disc_U = UNetDiscriminator(1, 1)
    out_enc, fmaps_enc, out_dec = disc_U(x_add)
    print(f"input.size() = {x_add.size()}\n")

    print("<encoder>")
    print(f"out_enc.size() = {out_enc.size()}\n")

    print("<fmaps_enc>")
    print(f"len(fmaps_enc) = {len(fmaps_enc)}")
    for idx, fmap in enumerate(fmaps_enc):
        print(f"fmaps_enc.size()[{idx}] = {fmap.size()}")
    
    print("")
    print("<decoder>")
    print(f"out_dec.size() = {out_dec.size()}")
    


if __name__ == "__main__":
    main()