import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import omegaconf


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BART(nn.Module):
    def __init__(
        self,
        cfg: omegaconf.dictconfig.DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(1000, 512, padding_idx=0)
        self.posenc = PositionalEncoding(
            d_model=512,
            dropout=0.1,
            max_len=1000,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=512 * 4,
                dropout=0.1,
                activation='relu',
            ),
            num_layers=6,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=512 * 4,
                dropout=0.1,
                activation='relu',
            ),
            num_layers=6,
        )
        self.output_layer = nn.Linear(512, 1000)
        
    def forward(
        self,
        src_text: torch.Tensor,
        src_text_len: torch.Tensor,
        tgt_text: Optional[torch.Tensor],
        tgt_text_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        '''
        Arguments:
            src_text: (B, T)
            src_text_len: (B,)
            tgt_text: (B, T), target text must be shifted during training.
            tgt_text_len: (B,)
            
        Returns:
            decoder_output: (T, B, C)
        '''
        src_text = self.embedding(src_text) * math.sqrt(512)
        src_text = self.posenc(src_text)
        src_text = src_text.permute(1, 0, 2)    # (T, B, C)
        tgt_text = self.embedding(tgt_text) * math.sqrt(512)
        tgt_text = self.posenc(tgt_text)
        tgt_text = tgt_text.permute(1, 0, 2)    # (T, B, C)
        
        src_padding_mask = torch.arange(src_text.shape[0]).unsqueeze(0).expand(src_text.shape[1], -1)
        src_padding_mask = src_padding_mask > src_text_len.unsqueeze(-1)
        tgt_padding_mask = torch.arange(tgt_text.shape[0]).unsqueeze(0).expand(tgt_text.shape[1], -1)
        tgt_padding_mask = tgt_padding_mask > tgt_text_len.unsqueeze(-1)
        causal_mask = torch.triu(torch.full((tgt_text.shape[0], tgt_text.shape[0]), True, device=tgt_text.device), diagonal=1)
        
        encoder_output = self.encoder(
            src=src_text,
            src_key_padding_mask=src_padding_mask,
        )
        decoder_output = self.decoder(
            tgt=tgt_text,
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        output = self.output_layer(decoder_output)
        return output
    
    
import hydra
import random
@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg):
    bart = BART(cfg)
    batch_size = 16
    min_len = 25
    enc_max_len = 100
    dec_max_len = 150
    
    src_text_list = []
    src_text_len_list = []
    tgt_text_list = []
    tgt_text_len_list = []
    for i in range(batch_size):
        src_text = torch.randint(1, 1000, (random.randint(min_len, enc_max_len),))
        src_text_len_list.append(src_text.shape[0])
        tgt_text = torch.randint(1, 1000, (random.randint(min_len, dec_max_len),))
        tgt_text_len_list.append(tgt_text.shape[0])
        src_text = F.pad(src_text, (0, enc_max_len - src_text.shape[0]), 'constant', 0)
        tgt_text = F.pad(tgt_text, (0, dec_max_len - tgt_text.shape[0]), 'constant', 0)
        src_text_list.append(src_text)
        tgt_text_list.append(tgt_text)
    
    src_text = torch.stack(src_text_list, dim=0)
    src_text_len = torch.tensor(src_text_len_list)
    tgt_text = torch.stack(tgt_text_list, dim=0)
    tgt_text_len = torch.tensor(tgt_text_len_list)
    
    tgt_text_pred = bart(
        src_text=src_text,
        src_text_len=src_text_len,
        tgt_text=tgt_text,
        tgt_text_len=tgt_text_len,
    )
    
    
if __name__ == '__main__':
    main()