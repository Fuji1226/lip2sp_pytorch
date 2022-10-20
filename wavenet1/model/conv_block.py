import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .conv import CausalConv1d


class ResSkipBlock(nn.Module):
    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels, dilation, feature_channels, dropout=0):
        super().__init__()
        self.causal_conv = CausalConv1d(
            in_channels=residual_channels,
            out_channels=gate_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            apply_weight_norm=False,
        )
        self.conv_feature = nn.Conv1d(feature_channels, gate_channels, kernel_size=1, bias=False)
        self.conv_out = nn.Conv1d(gate_channels // 2, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(gate_channels // 2, skip_out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, feature):
        return self._forward(x, feature, False)

    def incremental_forward(self, x, feature):
        return self._forward(x, feature, True)

    def _forward(self, x, feature, incremental):
        """
        x : (B, C, T)
        feature : (B, C, T)
        """
        res = x
        x = self.dropout(x)

        if incremental:
            split_dim = 1
            y = self.causal_conv.incremental_forward(x)     # (B, C, T)
        else:
            split_dim = 1
            y = self.causal_conv(x)

        # チャンネル方向に2分割
        y1, y2 = torch.split(y, y.shape[split_dim] // 2, dim=split_dim)

        # local condition
        feature = self.conv_feature(feature)
        feature1, feature2 = torch.split(feature, feature.shape[split_dim] // 2, dim=split_dim)
        y1 = y1 + feature1
        y2 = y2 + feature2

        # gated activation
        y = torch.tanh(y1) * torch.sigmoid(y2)

        # スキップ接続用の出力
        skip = self.conv_skip(y)

        # 出力
        out = self.conv_out(y)
        out += res

        return out, skip
        
    def clear_buffer(self):
        self.causal_conv.clear_buffer()
        

