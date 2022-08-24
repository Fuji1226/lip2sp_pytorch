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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

try:
    from .embedding import PositionalEncoding
    from .modules import Linear
    from ..transformer import make_pad_mask
except:
    from embedding import PositionalEncoding
    from modules import Linear
    from transformer import make_pad_mask


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)    # (B, T, n_head, C // n_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # (B, n_head, T, C // n_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)    # (B, n_head, T, C // n_head)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)  # (B, T, C) -> (B, T, n_head, C // n_head)
        
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))    # (B, n_head, T, T)

        # relative positional encoding(transformer-XL)
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))  # (B, n_head, T, T)
        pos_score = self._relative_shift(pos_score)
        
        score = (content_score + pos_score) / self.sqrt_dim
        
        if mask is not None:
            mask = mask.unsqueeze(1)    
            score.masked_fill_(mask, torch.tensor(float('-inf')))     # maskがtrueのところを-infに変更

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)     # (B, T, n_head, C // n_head)
        context = context.contiguous().view(batch_size, -1, self.d_model)   # (B, T, C) ヘッドごとの出力を結合

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)  # (B, n_head, T, 1)

        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)    # (B, n_head, T, T + 1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)   # (B, n_head, T + 1, T)
        
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)   # (B, n_head, T, T)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs, mask=None):
        """
        inputs : (B, T, C)
        mask : (B, 1, T)

        return 
        outputs : (B, T, C)
        """
        batch_size, seq_length, _ = inputs.size()

        # positional embedding
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)  # (1, T, C) -> (B, T, C)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)



def main():
    d_model = 256
    n_head = 8
    relative_atten = RelativeMultiHeadAttention(d_model=d_model, num_heads=n_head)
    attention = MultiHeadedSelfAttentionModule(d_model, n_head)

    # data_len
    data_len = [300, 300, 300, 300, 100, 100, 200, 200]
    data_len = torch.tensor(data_len)
    max_len = 300 // 2
    reduction_factor = 2
    data_len = torch.div(data_len, reduction_factor)
    mask = make_pad_mask(data_len, max_len)

    batch = 8
    frames = 150
    lip = torch.rand(batch, d_model, frames)
    lip = lip.permute(0, -1, -2)    # (B, T, C)
    
    out = attention(lip, mask)

    return


if __name__ == "__main__":
    main()