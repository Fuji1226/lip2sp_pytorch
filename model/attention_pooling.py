import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_posenc(x, start_index=0):
    """
    x : (B, T, H, W, C)
    """
    B, T, H, W, C = x.shape

    depth = np.arange(C) // 2 * 2
    depth = np.power(10000.0, depth / C)
    pos = np.arange(start_index, start_index + (H * W))
    phase = pos[:, None] / depth[None]

    phase[:, ::2] += float(np.pi/2)
    positional_encoding = np.sin(phase)

    positional_encoding = positional_encoding.T[None]
    positional_encoding = torch.from_numpy(positional_encoding).to(x.device)
    positional_encoding = positional_encoding.to(torch.float32)
    return positional_encoding      # (1, C, H * W)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.temperature = (d_model // n_head) ** 0.5
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """
        x : (B, T, H, W, C)
        """
        res = x
        B, T, H, W, C = x.shape
        q = self.w_q(x).reshape(B, T, H, W, self.n_head, -1).permute(0, 4, 1, 2, 3, 5)   # (B, head, T, H, W, C)
        k = self.w_k(x).reshape(B, T, H, W, self.n_head, -1).permute(0, 4, 1, 2, 3, 5)   # (B, head, T, H, W, C)
        v = self.w_v(x).reshape(B, T, H, W, self.n_head, -1).permute(0, 4, 1, 2, 3, 5)   # (B, head, T, H, W, C)

        q = q.reshape(B, self.n_head, T, H * W, -1)
        k = k.reshape(B, self.n_head, T, H * W, -1).permute(0, 1, 2, 4, 3)
        v = v.reshape(B, self.n_head, T, H * W, -1)

        attention = torch.matmul(q / self.temperature, k)   # (B, head, T, H * W, H * W)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)     
        output = torch.matmul(attention, v)     # (B, head, T, H * W, C)
        output = output.reshape(B, self.n_head, T, H, W, -1).permute(0, 2, 3, 4, 1, 5)      # (B, T, H, W, head, C)
        output = output.reshape(B, T, H, W, C)
        output = self.fc(output)
        output = output + res
        output = self.layer_norm(output)
        return output   # (B, T, H, W, C)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.layer_norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : (B, T, H, W, C)
        """
        res = x
        output = self.fc2(F.relu(self.fc1(x)))
        output = self.dropout(output)
        output = output + res
        output = self.layer_norm(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, d_model)
        self.fc = PositionwiseFeedForward(d_model, d_model * 4)

    def forward(self, x):
        """
        x : (B, T, H, W, C)
        """
        x = self.attention(x)
        x = self.fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, n_head, d_model, dropout=0.1):
        super().__init__()
        layers = []
        self.dropout = nn.Dropout(dropout)
        for _ in range(n_layers):
            layers.append(EncoderLayer(n_head, d_model))
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """
        x : (B, T, H, W, C)
        """ 
        B, T, H, W, C = x.shape
        x = self.dropout(x)
        pos = spatial_posenc(x)     # (B, C, H * W)
        pos = pos.reshape(1, C, H, W).unsqueeze(1)  # (1, 1, C, H, W)
        pos = pos.permute(0, 1, 3, 4, 2)    # (1, 1, H, W, C)
        x = x + pos
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    net = Encoder(2, 2, 128)
    x = torch.rand(1, 150, 6, 6, 128)
    out = net(x)
    print(out.shape)