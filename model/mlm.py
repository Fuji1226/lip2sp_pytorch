import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))
import math
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from .transformer_remake import EncoderLayer, posenc
    from .taco import PreNet
except:
    from transformer_remake import EncoderLayer, posenc
    from taco import PreNet


class MLMTrainer(nn.Module):
    def __init__(self, n_code=100, n_layers=1, n_head=4, d_model=256, reduction_factor=1, dropout=0.1) -> None:
        super().__init__()
        self.mask_prob = 0.15
        self.replace_prob = 0.9

        self.num_tokens = 100
        self.random_token_prob = 0.15

        
        self.mask_ignore_token_ids = None
        self.pad_token_id = n_code + 1
        self.mask_token_id = n_code
        self.mask_ignore_token_ids = set([self.mask_ignore_token_ids, self.pad_token_id])
        
        
        self.encoder = MLM(n_code, n_layers, n_head, d_model, reduction_factor, dropout)
        self.last = nn.Linear(d_model, n_code)
        
    def forward(self, seq, data_len, device):
        seq = torch.where(seq == -1, torch.tensor(self.pad_token_id), seq)
    
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)
        
        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_seq = seq.clone().detach()

        # derive labels to predict
        labels = seq.masked_fill(~mask, self.pad_token_id)
        
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # remove tokens that were substituted randomly from being [mask]ed later
            mask = mask & ~random_token_prob
            
        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)
        
        masked_seq = masked_seq.to(device)
        data_len = data_len.to(device)
        labels = labels.to(device)
        
        logit = self.encoder(masked_seq, data_len)
        logit = self.last(logit)
        
        mlm_loss = F.cross_entropy(
            logit.transpose(1, 2),
            labels,
            ignore_index = self.pad_token_id
        )
        return mlm_loss
            
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob
        
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


class MLM(nn.Module):
    def __init__(self, n_code, n_layers, n_head, d_model, reduction_factor, dropout=0.1) -> None:
        super().__init__()
        
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_inner = d_model * 4
        self.reduction_factor = reduction_factor
        
        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)
        
        self.emb = nn.Embedding(n_code+2, d_model)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, self.d_inner, n_head, self.d_k, self.d_v, dropout)
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, x, data_len):  
        max_len = int(max(data_len))
        mask = make_pad_mask(data_len, max_len)
        mask = mask.repeat(1, 1, max_len).transpose(1, 2)

        x = self.emb(x)
        x = self.dropout(x)
       
        #x = x + posenc(x, device=x.device, start_index=0)
        pos =positional_encoding(seq_len=x.shape[1], d_model=x.shape[2])
        pos = pos.repeat(x.shape[0], 1, 1).to(x.device)
        x = x + pos
        x = x.float()
 
        enc_output = self.layer_norm(x)

        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, mask)

        return enc_output


def positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    pe = torch.tensor(pe)
    return pe

def make_pad_mask(lengths, max_len):
    """
    口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
    """
    # この後の処理でリストになるので先にdeviceを取得しておく
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths)

    mask = seq_range_expand >= seq_length_expand
    mask = mask.unsqueeze(-1).to(device=device)
    
    return mask
    
if __name__=='__main__':
    data = torch.randint(0, 100, (8, 100))
    x_len = torch.tensor([100, 100, 100, 100, 100, 80, 80, 80])

    trainer = MLMTrainer()
    trainer(data, x_len)