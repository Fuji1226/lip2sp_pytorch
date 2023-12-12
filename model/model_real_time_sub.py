import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import CausalResNet3D


class ZoneOutCellGRU(nn.Module):
    def __init__(self, cell, zoneout):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, input, hidden):
        next_hidden = self.cell(input, hidden)
        next_hidden = self._zoneout(hidden, next_hidden)
        return next_hidden

    def _zoneout(self, h, next_h):
        next_h = self._apply_zoneout(h, next_h)
        return next_h

    def _apply_zoneout(self, h, next_h):
        if self.training:
            mask = h.new(*h.size()).bernoulli_(self.zoneout)
            return mask * h + (1 - mask) * next_h
        else:
            return self.zoneout * h + (1 - self.zoneout) * next_h


class Lip2SPRealTimeS(nn.Module):
    def __init__(
        self, in_channels, res_inner_channels, res_dropout,
        n_gru_layers, zoneout, out_channels, reduction_factor):
        super().__init__()
        self.inner_channels = int(res_inner_channels * 8) #32*8
        self.out_channels = out_channels #80
        self.reduction_factor = reduction_factor #2
        self.resnet = CausalResNet3D(
            in_channels=in_channels,
            out_channels=self.inner_channels,
            inner_channels=res_inner_channels,
            dropout=res_dropout,
        )

        gru = []
        for i in range(n_gru_layers):
            gru.append(ZoneOutCellGRU(
                nn.GRUCell(self.inner_channels, self.inner_channels), zoneout,
            ))
        self.gru = nn.ModuleList(gru)
        self.out_layer = nn.Linear(self.inner_channels, out_channels * reduction_factor)

def forward(self, lip, spk_emb=None, num_frames=None):
        res_output = self.resnet(lip)   # (B, C, T)
        batch_size = res_output.size(0)

        h_list = []
        for i in range(len(self.gru)):
            h_list.append(torch.zeros(batch_size, self.inner_channels).to(res_output.device, res_output.dtype))

        output_list = []
        if num_frames is None:
            num_frames = res_output.shape[-1]

        for i in range(num_frames):
            h_list[0] = self.gru[0](res_output[..., i], h_list[0])
            for j in range(1, len(self.gru)):
                h_list[j] = self.gru[j](h_list[j - 1], h_list[j])

            output = self.out_layer(h_list[-1])
            output = output.reshape(-1, self.out_channels, self.reduction_factor)
            output_list.append(output)

        output = torch.cat(output_list, dim=-1)     # (B, C, T)
        return output, None, None