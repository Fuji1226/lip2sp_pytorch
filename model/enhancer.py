"""
speech enhancementの利用
モデルから予測された音響特徴量を入力し,それを原音声の音響特徴量に近づけるための後処理ネットワークとしてやってみたい

Enhancer2Dは2次元畳み込み,1Dは1次元畳み込み
メルスペクトログラムに対してはどちらもできると思うけど,WORLDに2次元畳み込みはよくわからないので1Dがいいかなと思ってます
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Enhancer2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_lstm=False, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
            ),
        ])

        if use_lstm:
            self.lstm = nn.LSTM(1280, 1280, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
            if bidirectional:
                self.fc = nn.Linear(1280 * 2, 1280)     # 双方向性の時は順方向と逆方向の両方になるので次元が倍になる
            else:
                self.fc = nn.Linear(1280, 1280)

        self.trans_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256 * 2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128 * 2, 64, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64 * 2, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32 * 2, 16, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(16 * 2, out_channels, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
                nn.BatchNorm2d(num_features=1),
                nn.Softplus(),
            ),
        ])

    def forward(self, x):
        """
        x : (B, C, T)
        out : (B, C, T)
        """
        B, C, T = x.shape

        # 2次元畳み込みを行うので次元を増やす
        out = x.unsqueeze(1)    # (B, 1, C, T)
        out = out.permute(0, 1, 3, 2)   # (B, 1, T, C)
        
        # スキップ接続用に各層での出力を保持する
        skips = []

        # encoder layer
        for layer in self.conv_layers:
            out = layer(out)
            skips.append(out)

        if hasattr(self, "lstm"):
            out = out.permute(0, 2, 1, 3)  # (B, T, 256, C)
            out = out.contiguous().view(B, T, -1)    # (B, T, 256 * C)
            
            # lstm
            out, _ = self.lstm(out)
            out = self.fc(out)
            out = out.contiguous().view(B, T, 256, -1)   # (B, T, 256, C)
            out = out.permute(0, 2, 1, 3)  # (B, 256, T, C)

        # 反転
        skips_reverse = reversed(skips)

        # decoder layer
        for skip, layer in zip(skips_reverse, self.trans_conv_layers):
            out = torch.cat([out, skip], dim=1)
            out = layer(out)
            
        out = out.squeeze(1)    # (B, T, C)
        out = out.permute(0, -1, -2)    # (B, C, T)
        return out


class Enhancer1D(nn.Module):
    def __init__(self, in_channels, out_channels, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.conv_layers1 = [
            nn.Sequential(
                nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ),
        ]

        self.lstm = nn.LSTM(512, 512, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(512 * 2, 512)   # 双方向性の時は順方向と逆方向の両方になるので次元が倍になる
        else:
            self.fc = nn.Linear(512, 512)

        self.conv_layers2 = [
            nn.Sequential(
                nn.Conv1d(512 * 2, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(256 * 2, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(128 * 2, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.Softplus(),
            ),
        ]

    def forward(self, x):
        """
        x : (B, C, T)
        out : (B, C, T)
        """
        out = x
        skips = []

        for layer in self.conv_layers1:
            out = layer(out)
            skips.append(out)

        out = out.permute(0, -1, -2)    # (B, T, C)
        out, _ = self.lstm(out)
        out = self.fc(out)
        out = out.permute(0, -1, -2)    # (B, C, T)

        skips = reversed(skips)

        for skip, layer in zip(skips, self.conv_layers2):
            out = torch.cat([out, skip], dim=1)
            out = layer(out)

        return out



def main():
    B = 16
    C = 80
    T = 300
    x = torch.rand(B, C, T)
    net = Enhancer2D()
    out = net(x)
    print(out.shape)


if __name__ == "__main__":
    main()