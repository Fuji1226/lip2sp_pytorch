import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, bidirectional, dropout, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, data_len=None):
        """
        x : (B, C, T)
        """
        B, C, T = x.shape
        x = self.dropout(x.permute(0, 2, 1))    # (B, T, C)

        if data_len is not None:
            data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
            data_len = torch.clamp(data_len, min=min(data_len), max=T)
            x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)

        out, (hn, cn) = self.lstm(x)

        if data_len is not None:
            out = pad_packed_sequence(out, batch_first=True)[0]

        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


class GRUEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, bidirectional, dropout, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_channels, hidden_channels, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, data_len=None):
        """
        x : (B, C, T)
        """
        B, C, T = x.shape
        x = self.dropout(x.permute(0, 2, 1))    # (B, T, C)

        if data_len is not None:
            data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
            data_len = torch.clamp(data_len, min=min(data_len), max=T)
            x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)

        out, hn = self.gru(x)

        if data_len is not None:
            out = pad_packed_sequence(out, batch_first=True)[0]
            if out.shape[1] < T:
                zero_pad = torch.zeros(out.shape[0], T - out.shape[1], out.shape[2]).to(device=out.device, dtype=out.dtype)
                out = torch.cat([out, zero_pad], dim=1)

        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


if __name__ == "__main__":
    batch = 2
    hid = 128
    net = LSTMEncoder(hid, 1, bidirectional=True, dropout=0.1, reduction_factor=2)
    x = torch.rand(batch, hid, 150)
    data_len = torch.tensor([x.shape[-1] * 2, x.shape[-1]])
    out = net(x, data_len)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    net = GRUEncoder(hid, 1, bidirectional=True, dropout=0.1, reduction_factor=2)
    x = torch.rand(batch, hid, 150)
    data_len = torch.tensor([x.shape[-1] * 2, x.shape[-1]])
    out = net(x, data_len)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")
    
    x = torch.rand(3, 2, 10).permute(0, 2, 1)
    print(f"x = {x.shape}")
    data_len = torch.tensor([5, 5, 8])
    data_len = torch.clamp(data_len, min=min(data_len), max=x.shape[1])
    x_pack = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
    x_recon = pad_packed_sequence(x_pack, batch_first=True)[0]
    zero_pad = torch.zeros(x_recon.shape[0], x.shape[1] - x_recon.shape[1], x_recon.shape[-1])
    x_recon = torch.cat([x_recon, zero_pad], dim=1)
    # print(f"x_pack = {x_pack}")
    print(f"x_recon = {x_recon.shape}")
    print(f"x = {x}")
    print(f"x_recon = {x_recon}")