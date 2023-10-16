import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerClassifierRNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layers, bidirectional, n_speaker):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.last_layer = nn.Linear(hidden_dim, n_speaker)

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


class SpeakerClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_speaker):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.last_layer = nn.Linear(hidden_channels, n_speaker)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        out = torch.mean(x, dim=-1)     # (B, C)
        out = self.layer(out)
        out = self.last_layer(out)
        return out