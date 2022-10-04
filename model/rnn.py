import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, data_len=None):
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


class GRUEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, bidirectional):
        super().__init__()
        self.gru = nn.GRU(hidden_channels, hidden_channels, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, data_len=None):
        x = x.permute(0, 2, 1)
        out, hn = self.gru(x)
        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


if __name__ == "__main__":
    hid = 128
    net = LSTMEncoder(hid, 1, bidirectional=True)
    x = torch.rand(1, hid, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    net = GRUEncoder(hid, 1, bidirectional=True)
    x = torch.rand(1, hid, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")