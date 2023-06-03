import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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
            nn.Conv1d(in_channels, hidden_channels, bias=False, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, bias=False, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.last_layer = nn.Linear(hidden_channels, n_speaker)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = self.layer(x)
        x = torch.mean(x, dim=-1)     # (B, C)
        x = self.last_layer(x)
        return x
    
    
class RecordedSynthClassifierSpeech(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_conv_layers, lstm_n_layers):
        super().__init__()
        convs = []
        for i in range(n_conv_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            convs.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ))    
        self.convs = nn.ModuleList(convs)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels // 2, num_layers=lstm_n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, data_len):
        """
        Args:
            x (_type_): (B, C, T)
        """
        for conv in self.convs:
            x = conv(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        B, T, C = x.shape
        
        data_len = torch.clamp(data_len,  max=T)
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)    
        x = pad_packed_sequence(x, batch_first=True)[0]
        if x.shape[1] < T:
            zero_pad = torch.zeros(x.shape[0], T - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)
        
        x = torch.mean(x, dim=1)    # (B, C)
        x = self.fc(x)
        return x
    
    
class RecordedSynthClassifierVideo(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_conv_layers, lstm_n_layers):
        super().__init__()
        convs = []
        for i in range(n_conv_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            convs.append(nn.Sequential(
                nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=(2, 2, 1)),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ))
        self.convs = nn.ModuleList(convs)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels // 2, num_layers=lstm_n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, data_len):
        """
        Args:
            x (_type_): (B, C, H, W, T)
            data_len (_type_): _description_

        Returns:
            _type_: _description_
        """
        for conv in self.convs:
            x = conv(x)
        x = torch.mean(x, dim=(2, 3))   # (B, C, T)
        x = x.permute(0, 2, 1)      # (B, T, C)
        B, T, C = x.shape
        
        data_len = torch.clamp(data_len, max=T)
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)    
        x = pad_packed_sequence(x, batch_first=True)[0]
        if x.shape[1] < T:
            zero_pad = torch.zeros(x.shape[0], T - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)
        
        x = torch.mean(x, dim=1)    # (B, C)
        x = self.fc(x)
        return x