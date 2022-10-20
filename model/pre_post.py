
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prenet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=32, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(inner_channels, inner_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(inner_channels, out_channels, 1),
            nn.ReLU()
        )

        # self.conv1 = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        # self.conv2 = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        # self.conv3 = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

        self.project_pre = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.layer_norm = nn.LayerNorm(out_channels)

        self.dropout = dropout
        

    def forward(self, x):
        """
        音響特徴量をtransformer内の次元に調整する役割
        x : (B, C=feature channels, T)
        y : (B, C=d_model, T)
        """
        # out = F.relu(self.conv1(x))
        # out = F.dropout(out, self.dropout, training=True)

        # out = F.relu(self.conv2(out))
        # out = F.dropout(out, self.dropout)

        # out = F.relu(self.conv3(out))
        out = self.fc(x)

        out = self.project_pre(out)
        out = self.layer_norm(out.transpose(-1, -2))

        #breakpoint()
        return out.transpose(-1, -2)


class Postnet(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, n_layers=5, dropout=0.5, kernel_size=9):
        super().__init__()
        self.padding = kernel_size -1

        conv = []
        conv.append(nn.Conv1d(in_channels, inner_channels, kernel_size=kernel_size, padding=self.padding, bias=True))
        conv.append(nn.BatchNorm1d(inner_channels))
        conv.append(nn.Tanh())
        conv.append(nn.Dropout(p=dropout))
        
        for _ in range(n_layers - 2):
            conv.append(nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=self.padding, bias=True))
            conv.append(nn.BatchNorm1d(inner_channels))
            conv.append(nn.Tanh())
            conv.append(nn.Dropout(p=dropout))

        conv.append(nn.Conv1d(inner_channels, out_channels, kernel_size=kernel_size, padding=self.padding, bias=True))
        self.conv_layers = nn.ModuleList(conv)

    def forward(self, x):
        y = x 
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                y = layer(y)[..., :-self.padding]
            else:
                y = layer(y)
        return x + y


class ProgressivePrenet(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_factor, inner_channels=32, dropout=0.5):
        super().__init__()
        assert in_channels // reduction_factor == 40 or 60 or 80
        self.in_channels = in_channels
        self.reduction_factor = reduction_factor

        self.conv40 = nn.Conv1d(40 * reduction_factor, inner_channels, kernel_size=1)
        self.conv60 = nn.Conv1d(60 * reduction_factor, inner_channels, kernel_size=1)
        self.conv80 = nn.Conv1d(80 * reduction_factor, inner_channels, kernel_size=1)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(inner_channels, out_channels, kernel_size=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        """
        音響特徴量をtransformer内の次元に調整する役割
        in_channelsによって最初の層を変更

        x : (B, C=feature channels, T)
        out : (B, C=d_model, T)
        """
        out = x

        if self.in_channels == 40 * self.reduction_factor:
            out = self.conv40(out)
        elif self.in_channels == 60 * self.reduction_factor:
            out = self.conv60(out)
        elif self.in_channels == 80 * self.reduction_factor:
            out = self.conv80(out)

        out = self.layers(out)
        return out


class ProgressivePostnet(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, n_layers=5, dropout=0.5):
        super().__init__()
        assert in_channels == 40 or 60 or 80
        self.in_channels = in_channels

        self.conv40_in = nn.Conv1d(40, inner_channels, kernel_size=5, padding=2, bias=True)
        self.conv60_in = nn.Conv1d(60, inner_channels, kernel_size=5, padding=2, bias=True)
        self.conv80_in = nn.Conv1d(80, inner_channels, kernel_size=5, padding=2, bias=True)
        self.conv_layers = nn.Sequential(
            nn.BatchNorm1d(inner_channels),
            nn.Tanh(),
            nn.Dropout(p=dropout)
        )

        for _ in range(n_layers - 2):
            self.conv_layers.append(nn.Conv1d(inner_channels, inner_channels, kernel_size=5, padding=2, bias=True))
            self.conv_layers.append(nn.BatchNorm1d(inner_channels))
            self.conv_layers.append(nn.Tanh())
            self.conv_layers.append(nn.Dropout(p=dropout))

        self.conv40_out = nn.Conv1d(inner_channels, 40, kernel_size=5, padding=2, bias=True)
        self.conv60_out = nn.Conv1d(inner_channels, 60, kernel_size=5, padding=2, bias=True)
        self.conv80_out = nn.Conv1d(inner_channels, 80, kernel_size=5, padding=2, bias=True)

    def forward(self, x):
        residual = x
        out = x

        if self.in_channels == 40:
            out = self.conv40_in(out)
        elif self.in_channels == 60:
            out = self.conv60_in(out)
        elif self.in_channels == 80:
            out = self.conv80_in(out)
        
        out = self.conv_layers(out)

        if self.in_channels == 40:
            out = self.conv40_out(out)
        elif self.in_channels == 60:
            out = self.conv60_out(out)
        elif self.in_channels == 80:
            out = self.conv80_out(out)

        out = residual + out
        return out
    
def main():
    B = 8
    C = 80
    T = 300
    x = torch.rand(B, C, T)
    prenet = ProgressivePrenet(C, 256)
    # print(prenet)
    out= prenet(x)

    postnet = ProgressivePostnet(C, 512, C)
    # print(postnet)
    out = postnet(x)


if __name__ == "__main__":
    main()