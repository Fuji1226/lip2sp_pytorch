import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layers, bidirectional, n_speaker):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, 256)
        else:
            self.fc = nn.Linear(hidden_dim, 256)
        
        self.last_layer = nn.Linear(256, n_speaker)

    def forward(self, x):
        """
        x : (B, T, C)
        out : (B, C)
        """
        out, (hn, cn) = self.lstm(x)
        out = torch.cat([hn[-1], hn[-2]], dim=-1)
        out = self.fc(out)
        out = self.last_layer(out)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 10, 1)
    net = SpeakerClassifier(1, 4, 2, True, n_speaker=2)
    out = net(x)