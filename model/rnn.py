import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, n_layers, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, out_channels)
        else:
            self.fc = nn.Linear(hidden_dim, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, data_len=None):
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


class GRUEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, n_layers, bidirectional):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, out_channels)
        else:
            self.fc = nn.Linear(hidden_dim, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, data_len=None):
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.gru(x)
        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


if __name__ == "__main__":
    net = LSTMEncoder(256, 256, 128, 1, bidirectional=True)
    x = torch.rand(1, 256, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    net = GRUEncoder(256, 256, 128, 1, bidirectional=True)
    x = torch.rand(1, 256, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")