import torch 
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

# 複数話者
class JCUDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, use_gc=True, emb_in=None, n_features=128, out_features=512):
        super().__init__()
        self.use_gc = use_gc

        # 共有する層
        self.share = [
            nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(128, 512, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ),
        ]
        self.share = nn.ModuleList(self.share)

        # unconditional layers
        self.uncond = nn.Sequential(
                nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2),
            )

        # conditional layers
        if self.use_gc:
            self.embed = nn.Embedding(emb_in, n_features)
            self.fc = nn.Linear(n_features, out_features)
            self.cond = nn.Sequential(
                    nn.Conv1d(512 + out_features, 128, kernel_size=5, stride=1, padding=2),
                    nn.LeakyReLU(0.2),
                )

        # 出力層
        self.out_layer = nn.Sequential(
                nn.Conv1d(128, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x, gc=None):
        """
        x : (B, mel_channels, t)
        gc : (B, 1)
        """
        if self.use_gc:
            assert gc is not None, "please set global condition"
        else:
            assert gc is None, "Don't set global condition"

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
        out_uncond = self.uncond(out_uncond)
        fmaps_uncond.append(out_uncond)
        out_uncond = self.out_layer(out_uncond)
        
        # conditional output
        if gc is not None:
            gc = self.embed(gc)   # (B, 1) -> (B, 1, n_features)
            gc = self.fc(gc)      # (B, 1, n_features) -> (B, 1, 512)
            gc = torch.transpose(gc, 1, 2)    # (B, 1, 512) -> (B, 512, 1)

            # 論文ではconditionを時間方向に拡大すると書いてあったけど、expandを使うということなのかよくわからない
            # 一応VocGANではconditionであるメルスペクトログラムを時間方向に結合していた
            gc = gc.expand(out_cond.shape)  # 時間方向に拡大
            out_cond = torch.cat([out_cond, gc], dim=1)      # out_condとconditionをチャンネル方向に結合　512 -> 1024

            out_cond = self.cond(out_cond)
            fmaps_cond.append(out_cond)
            out_cond = self.out_layer(out_cond)

            return [out_uncond, out_cond], fmaps_share + fmaps_uncond + fmaps_cond

        return [out_uncond], fmaps_share + fmaps_uncond


class UNetDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        """
        原論文が単話者なので、一旦単話者を想定して実装
        encoderの特徴量の結合の仕方が論文に詳しく書いていないので微妙。とりあえずUNetと同様に、チャンネル方向に結合している。
        複数話者にも適応したい
        """
        self.encoder = [
            nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)),
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

            # 学習に使うフレーム数が300だと、一個前の特徴マップがt=75なので、半分にできなくてdecoder側と合わないのでバグる
            nn.Sequential(
                weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            ),
        ]
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
        ]
        self.decoder_out = nn.Sequential(
                weight_norm(nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        """
        x : (B, mel_channels, t)
        """
        x = x.unsqueeze(1)  # (B, mel_channels, t) -> (B, 1, mel_channels, t)
        fmaps_enc = []
        fmaps_dec = []

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
                fmaps_dec.append(out_dec)
            else:
                out_dec = torch.cat([out_dec, fmaps_enc[-1-idx]], dim=1)    # とりあえずUNetと同様にチャンネル方向に結合している。原論文に詳しく書いていないので一緒かはわからない。
                out_dec = layer(out_dec)
                fmaps_dec.append(out_dec)

        out_dec = torch.cat([out_dec, fmaps_enc[0]], dim=1)
        out_dec = self.decoder_out(out_dec)
        
        return [out_enc, out_dec], fmaps_enc + fmaps_dec


def main():
    B = 8
    mel_channels = 80
    t =400
    global_feature = torch.arange(B).reshape(B, 1)      # 複数話者の場合を想定
    x = torch.rand(B, mel_channels, t)
    use_gc = False
    
    # GANSpeechの場合
    print("## GANSpeech ##")
    GANSpeech = JCUDiscriminator(in_channels=mel_channels, out_channels=1, use_gc=use_gc, emb_in=B, )
    out, fmaps = GANSpeech(x)
    # out, fmaps = GANSpeech(x, global_feature)

    for i in range(len(fmaps)):
        print(f"fmaps[{i}] = {fmaps[i].shape}")

    # U-Netの場合
    print("## U-Net ##")
    disc_U = UNetDiscriminator(1, 1)
    out, fmaps = disc_U(x)

    for i in range(len(fmaps)):
        print(f"fmaps[{i}] = {fmaps[i].shape}")
    # for i in range(len(fmaps_enc)):
    #     print(f"fmaps_enc[{i}] = {fmaps_enc[i].shape}")
    # for i in range(len(fmaps_dec)):
    #     print(f"fmaps_dec[{i}] = {fmaps_dec[i].shape}")
    


if __name__ == "__main__":
    main()