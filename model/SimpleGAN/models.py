import torch
from nn import torch

class SimpleConv(nn.Module):
    def __init__(self, in_dim, out_dim=1, num_hidden=2, hidden_dim=256,
                 dropout=0.5, last_sigmoid=True):
        # bidirectional is dummy
        super().__init__()
        in_sizes = [in_dim] + [hidden_dim] * (num_hidden - 1)
        out_sizes = [hidden_dim] * num_hidden
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_size, out_size, kernel_size=3, padding=1) for (in_size, out_size)
             in zip(in_sizes, out_sizes)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.last_sigmoid = last_sigmoid

    def forward(self, x, lengths=None):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        x = self.last_linear(x)
        return self.sigmoid(x) if self.last_sigmoid else x