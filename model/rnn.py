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
            if out.shape[1] < T:
                zero_pad = torch.zeros(out.shape[0], T - out.shape[1], out.shape[2]).to(device=out.device, dtype=out.dtype)
                out = torch.cat([out, zero_pad], dim=1)

        out = self.fc(out)
        out = self.norm(out)
        return F.relu(out)


class GRUEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, dropout, reduction_factor, which_norm):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.which_norm = which_norm
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_channels, hidden_channels, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(int(hidden_channels * 2), hidden_channels)

        if which_norm == "ln":
            self.norm = nn.LayerNorm(hidden_channels)
        elif which_norm == "bn":
            self.norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        data_len : (B,)
        """
        B, C, T = x.shape
        x = self.dropout(x.permute(0, 2, 1))    # (B, T, C)

        # data_lenが全てTより大きい場合、pack_padded_sequenceでバグるのでclamp
        data_len = torch.clamp(data_len,  max=T)
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)

        out, hn = self.gru(x)

        out = pad_packed_sequence(out, batch_first=True)[0]
        if out.shape[1] < T:
            # data_lenが全てTより小さい場合、pad_packed_sequenceで系列長が短くなってしまうので0パディングで調整
            zero_pad = torch.zeros(out.shape[0], T - out.shape[1], out.shape[2]).to(device=out.device, dtype=out.dtype)
            out = torch.cat([out, zero_pad], dim=1)

        out = self.fc(out)

        if self.which_norm == "ln":
            out = self.norm(out)
        elif self.which_norm == "bn":
            out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        return F.relu(out)