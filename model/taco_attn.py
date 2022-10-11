import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_remake import make_pad_mask
from utils import count_params


class LocationSenstiveAttention(nn.Module):
    def __init__(self, enc_channels, dec_channels, hidden_channels, conv_channels, conv_kernel_size, reduction_factor):
        super().__init__()
        assert conv_kernel_size % 2 == 1
        self.reduction_factor = reduction_factor

        self.mlp_enc = nn.Linear(enc_channels, hidden_channels)
        self.mlp_dec = nn.Linear(dec_channels, hidden_channels, bias=False)
        self.mlp_att = nn.Linear(conv_channels, hidden_channels, bias=False)
        self.loc_conv = nn.Conv1d(1, conv_channels, conv_kernel_size, padding=(conv_kernel_size - 1) // 2, bias=False)
        self.w = nn.Linear(hidden_channels, 1)

        self.processed_memory = None

    def reset_state(self):
        self.processed_memory = None

    def forward(self, enc_output, dec_state, data_len, prev_att_w, mask=None):
        """
        enc_output : (B, T, C)
        dec_state : (B, C)
        data_len : (B,)
        prev_att_w : (B, T)
        mask : (B, T)

        att_w, att_c : (B, T)
        """
        B, T, C = enc_output.shape

        # encoder出力については毎回同じなのでインスタンスとして保持
        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(enc_output)    # (B, T, C)

        # 前時刻のアテンション重みがない場合は一様分布にする
        if prev_att_w is None:
            prev_att_w = 1.0 - make_pad_mask(data_len, T).squeeze(1).to(torch.int)
            prev_att_w /= data_len.unsqueeze(-1)    # (B, T)

        # location
        att_conv = self.loc_conv(prev_att_w.unsqueeze(1)).transpose(1, 2)   # (B, T, C)
        att_conv = self.mlp_att(att_conv)

        dec_state = self.mlp_dec(dec_state).unsqueeze(1)    # (B, 1, C)

        energy = torch.tanh(att_conv + self.processed_memory + dec_state)   # (B, T, C)
        energy = self.w(energy).squeeze(-1)     # (B, T)

        if mask is not None:
            energy.masked_fill_(mask, float("-inf"))

        # 時間方向にsoftmacを取ることで、現在のdec_stateとencoder出力全体との関連度が表現される
        att_w = F.softmax(energy, dim=-1)   # (B, T)

        # アテンション重みをチャンネル方向に拡大し、encoder出力との積和を計算
        att_c = torch.sum(enc_output * att_w.unsqueeze(-1), dim=1)  # (B, C)
        
        return att_c, att_w


if __name__ == "__main__":
    net = LocationSenstiveAttention(
        enc_channels=128,
        dec_channels=128,
        hidden_channels=128,
        conv_channels=128,
        conv_kernel_size=31,
        reduction_factor=2,
    )

    count_params(net, "net")    
