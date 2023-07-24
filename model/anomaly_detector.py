import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class AEDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, n_rnn_layers):
        super().__init__()
        convs_enc = []
        in_channels_list = [in_channels] + hidden_channels_list[:-1]
        for in_c, h_c in zip(in_channels_list, hidden_channels_list):
            convs_enc.append(
                nn.Sequential(
                    nn.Conv1d(in_c, h_c, kernel_size=3, padding=1),
                    nn.BatchNorm1d(h_c),
                    nn.ReLU(),
                )
            )
        self.convs_enc = nn.ModuleList(convs_enc)
        self.rnn = nn.LSTM(hidden_channels_list[-1], hidden_channels_list[-1] // 2, num_layers=n_rnn_layers, batch_first=True, bidirectional=True)

        convs_dec = []
        in_channels_list = list(reversed(hidden_channels_list))
        out_channels_list = in_channels_list[1:] + [in_channels]
        for in_c, out_c in zip(in_channels_list, out_channels_list):
            convs_dec.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(),
                )
            )
        self.convs_dec = nn.ModuleList(convs_dec)
        self.out_layer = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, data_len):
        """
        x : (B, C, T)
        """
        B, C, T = x.shape
        
        for conv in self.convs_enc:
            x = conv(x)
            
        x = x.permute(0, 2, 1)  # (B, T, C)
        data_len = torch.clamp(data_len,  max=T)
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        
        x, _ = self.rnn(x)
        
        x = pad_packed_sequence(x, batch_first=True)[0]
        if x.shape[1] < T:
            zero_pad = torch.zeros(x.shape[0], T - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)
        x = x.permute(0, 2, 1)  # (B, C, T)
        
        for conv in self.convs_dec:
            x = conv(x)
        x = self.out_layer(x)
        return x