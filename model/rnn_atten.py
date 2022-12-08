import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer_remake import make_pad_mask


class RNNAttention(nn.Module):
    def __init__(self, in_channels, dropout, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.conv = nn.Conv3d(in_channels, int(in_channels * 2), kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.mean_pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pooling = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.dropout  = nn.Dropout(dropout)
        self.gru = nn.GRU(int(in_channels * 4), int(in_channels * 4), num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(int(in_channels * 8), in_channels)

    def forward(self, x, data_len=None):
        """
        x : (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        x_conv = self.conv(x)
        x_mean = self.mean_pooling(x_conv).squeeze(-1).squeeze(-1)
        x_max = self.max_pooling(x_conv).squeeze(-1).squeeze(-1)
        att_w = torch.cat([x_mean, x_max], dim=1)  # (B, C, T)
        att_w = self.dropout(att_w.permute(0, 2, 1))  # (B, T, C)

        if data_len is not None:
            data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
            data_len = torch.clamp(data_len, min=min(data_len), max=T)
            att_w = pack_padded_sequence(att_w, data_len.cpu(), batch_first=True, enforce_sorted=False)

        att_w, hn = self.gru(att_w)

        if data_len is not None:
            att_w = pad_packed_sequence(att_w, batch_first=True)[0]
            if att_w.shape[1] < T:
                zero_pad = torch.zeros(att_w.shape[0], T - att_w.shape[1], att_w.shape[2]).to(device=att_w.device, dtype=att_w.dtype)
                att_w = torch.cat([att_w, zero_pad], dim=1)

        att_w = self.fc(att_w).permute(0, 2, 1)     # (B, C, T)
        if data_len is not None:
            mask = make_pad_mask(data_len, T)
            att_w = att_w.masked_fill(mask, torch.tensor(float('-inf')))

        att_w = torch.sigmoid(att_w).unsqueeze(-1).unsqueeze(-1)      # (B, C, T, 1, 1)
        return x * att_w


if __name__ == "__main__":
    net = RNNAttention(128, 0.1, 2)
    x = torch.rand(3, 128, 150, 12, 12)
    data_len = torch.tensor([150, 100, 200])
    out = net(x, data_len)
    print(out.shape)

    x = torch.tensor(float('-inf'))
    att_x = torch.sigmoid(x)
    print(x, att_x)

    x = torch.rand(1)
    att_x = torch.sigmoid(x)
    print(x, att_x)