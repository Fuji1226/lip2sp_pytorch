import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from pre_post import Prenet
from transformer_remake import make_pad_mask
from taco_attn import LocationSenstiveAttention
from utils import count_params


class RNNDecoder(nn.Module):
    def __init__(
        self, hidden_channels, out_channels, reduction_factor, pre_in_channels, pre_inner_channels,
         dropout, enc_channels, n_layers, conv_channels, conv_kernel_size, use_attention):
        super().__init__()
        self.out_channels = out_channels
        self.reduction_factor = reduction_factor
        self.hidden_channels = hidden_channels

        self.prenet = Prenet(pre_in_channels, hidden_channels, pre_inner_channels)
        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attention = LocationSenstiveAttention(
                enc_channels=enc_channels,
                dec_channels=hidden_channels,
                hidden_channels=hidden_channels,
                conv_channels=conv_channels,
                conv_kernel_size=conv_kernel_size,
                reduction_factor=reduction_factor,
            )

        gru = []
        for layer in range(n_layers):
            gru.append(
                nn.GRUCell(
                    enc_channels + hidden_channels if layer == 0 else hidden_channels,
                    hidden_channels,
                ),
            )
        self.gru = nn.ModuleList(gru)

        self.out_layer = nn.Conv1d(int(hidden_channels * 2), self.out_channels * self.reduction_factor, kernel_size=1)

    def forward(self, enc_output, data_len, target=None, training_method=None, threshold=None):
        """
        enc_output : (B, T, C)
        data_len : (B,)
        target(acoustic feature) : (B, C, T)

        out : (B, C, T)
        """
        B, T, C = enc_output.shape
        D = self.out_channels

        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)

        # view for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)  # (B, T, C)
            target = target.contiguous().view(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  # (B, C, T)
        else:
            target = torch.zeros(B, D * self.reduction_factor, 1).to(device=enc_output.device, dtype=enc_output.dtype) 

        h_list = []
        for _ in range(len(self.gru)):
            h_list.append(
                torch.zeros(B, self.hidden_channels).to(device=enc_output.device, dtype=enc_output.dtype)
            )

        go_frame = torch.zeros(B, int(D * 2), 1)
        prev_out = go_frame

        if hasattr(self, "attention"):
            prev_att_w = None
            self.attention.reset_state()

            data_len = torch.div(data_len, self.reduction_factor).to(dtype=torch.int)
            mask = make_pad_mask(data_len, T).squeeze(1)      # (B, T)

        out_list = []
        att_w_list = []
        t = 0
        while True:
            if hasattr(self, "attention"):
                att_c, att_w = self.attention(enc_output.permute(0, 2, 1), h_list[0], data_len, prev_att_w, mask)
                att_w_list.append(att_w)

            # prenet
            prenet_out = self.prenet(prev_out).squeeze(-1)    # (B, C)

            if hasattr(self, "attention"):
                rnn_input = torch.cat([prenet_out, att_c], dim=1)    
            else:
                rnn_input = torch.cat([prenet_out, enc_output[..., t]], dim=1)

            rnn_input = self.dropout(rnn_input)
            
            # rnn
            h_list[0] = self.gru[0](rnn_input, h_list[0])
            for i in range(1, len(self.gru)):
                h_list[i] = self.gru[i](h_list[i - 1], h_list[i])

            if hasattr(self, "attention"):
                hcs = torch.cat([h_list[-1], att_c], dim=-1)    # (B, C)
            else:
                hcs = torch.cat([h_list[-1], enc_output[..., t]], dim=-1)    # (B, C)

            out = self.out_layer(hcs.unsqueeze(-1))     # (B, C, 1)

            # update prev_out
            if training_method == "tf":
                prev_out = target[:, :, t].unsqueeze(-1)
                
            elif training_method == "ss":
                """
                threshold = 0 : teacher forcing
                threshold = 100 : using decoder prediction completely
                """
                rand = torch.randint(1, 101, (1,))
                if rand > threshold:
                    prev_out = out
                else:
                    prev_out = target[:, :, t].unsqueeze(-1)
            else:
                prev_out = out

            out_list.append(out.reshape(B, D, -1))

            # 累積アテンション重み
            if hasattr(self, "attention"):
                prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
            if t >= T:
                break

        # 時間方向に結合
        out = torch.cat(out_list, dim=-1)

        try:
            att_w = torch.stack(att_w_list, dim=-1)
        except:
            att_w = None

        return out, att_w


if __name__ == "__main__":
    data_len = torch.tensor([300, 200, 100])
    batch = data_len.shape[0]
    enc_output = torch.rand(batch, 150, 128)
    target = torch.rand(batch, 80, 300)

    net = RNNDecoder(
        hidden_channels=128,
        out_channels=target.shape[1],
        reduction_factor=2,
        pre_in_channels=int(target.shape[1] * 2),
        pre_inner_channels=32,
        dropout=0.1,
        enc_channels=enc_output.shape[-1],
        n_layers=2,
        conv_channels=32,
        conv_kernel_size=31,
        use_attention=True,
    )

    out, att_w = net(enc_output, data_len, target, training_method="ss", threshold=50)
    print(out.shape)
    if att_w is not None:
        print(att_w.shape)

    count_params(net, "net")