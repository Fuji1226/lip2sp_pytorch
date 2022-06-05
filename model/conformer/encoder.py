# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

try:
    from .feed_forward import FeedForwardModule
    from .attention import MultiHeadedSelfAttentionModule
    from .convolution import (
        ConformerConvModule,
        Conv2dSubampling,
    )
    from .modules import (
        ResidualConnectionModule,
        Linear,
    )
    from ..transformer import make_pad_mask
except:
    from feed_forward import FeedForwardModule
    from attention import MultiHeadedSelfAttentionModule
    from convolution import (
        ConformerConvModule,
        Conv2dSubampling,
    )
    from modules import (
        ResidualConnectionModule,
        Linear,
    )
    from transformer import make_pad_mask

from hparams import create_hparams


class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        # ResdualConnectionModuleは残差結合用のモジュール
        self.first_ff = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )

        self.attention = ResidualConnectionModule(
            module=MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
            ),
        )

        self.conv = ResidualConnectionModule(
            module=ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            ),
        )

        self.second_ff = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )

        self.layer_norm = nn.LayerNorm(encoder_dim)
        
    def forward(self, inputs: Tensor, mask: Tensor = None) -> Tensor:
        output = self.first_ff(inputs)
        output = self.attention(inputs, mask)
        output = self.conv(output)
        output = self.second_ff(output)
        return self.layer_norm(output)


class Conformer_Encoder(nn.Module):
    """
    encoder_dimとnum_layers、num_attention_heads以外は初期設定でOK
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        reduction_factor: int = 2,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=input_dropout_p)
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

        self.reduction_factor = reduction_factor
    
    def forward(self, prenet_out, data_len=None, max_len=None):
        """
        prenet_out : (B, C, T)

        return
        output : (B, T, C)
        """
        # get mask（学習時のみ、0パディングされた部分を隠すためのマスクを作成）
        if data_len is not None:
            assert max_len is not None
            data_len = torch.div(data_len, self.reduction_factor)
            mask = make_pad_mask(data_len, max_len)
        else:
            # 推論時はマスクなし
            mask = None
        
        output = prenet_out.permute(0, -1, -2)  # (B, T, C)
        # encoder layers
        for layer in self.layers:
            output = layer(output, mask)

        return output



def main():
     # parameter
    hparams = create_hparams()

    # data_len
    data_len = [300, 300, 200, 100]
    data_len = torch.tensor(data_len)
    max_len = hparams.length // 2

    batch = 4
    frames = 150
    lip_feature = torch.rand(batch, hparams.d_model, frames)
    con_enc = Conformer_Encoder(encoder_dim=hparams.d_model, num_layers=hparams.n_layers, num_attention_heads=hparams.n_head)
    out = con_enc(lip_feature, data_len, max_len)
    # out = con_enc(lip_feature)
    print(out.shape)
    return


if __name__ == "__main__":
    main()