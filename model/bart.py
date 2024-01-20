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
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.model.bart.n_vocab, cfg.model.bart.d_model, padding_idx=cfg.model.bart.padding_idx)
        self.posenc = PositionalEncoding(
            d_model=cfg.model.bart.d_model,
            dropout=cfg.model.bart.dropout,
            max_len=cfg.model.bart.phoneme_max_len * 4,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=cfg.model.bart.d_model,
                nhead=cfg.model.bart.nhead,
                dim_feedforward=cfg.model.bart.d_model * 4,
                dropout=cfg.model.bart.dropout,
                activation=cfg.model.bart.activation,
            ),
            num_layers=cfg.model.bart.num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=cfg.model.bart.d_model,
                nhead=cfg.model.bart.nhead,
                dim_feedforward=cfg.model.bart.d_model * 4,
                dropout=cfg.model.bart.dropout,
                activation=cfg.model.bart.activation,
            ),
            num_layers=cfg.model.bart.num_layers,
        )
        self.output_layer = nn.Linear(cfg.model.bart.d_model, cfg.model.bart.n_vocab)
        
    def forward(
        self,
        src_text: torch.Tensor,
        src_text_len: torch.Tensor,
        tgt_text: torch.Tensor,
        tgt_text_len: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Arguments:
            src_text: (B, T)
            src_text_len: (B,)
            tgt_text: (B, T)
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
        
        src_padding_mask = torch.arange(src_text.shape[0]).unsqueeze(0).expand(src_text.shape[1], -1).to(device=src_text.device)
        src_padding_mask = src_padding_mask > src_text_len.unsqueeze(-1)
        tgt_padding_mask = torch.arange(tgt_text.shape[0]).unsqueeze(0).expand(tgt_text.shape[1], -1).to(device=src_text.device)
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
    
    def greedy_search_validation(
        self,
        src_text: torch.Tensor,
        src_text_len: torch.Tensor,
        iter_limit: int,
        tgt_text: Optional[torch.Tensor],
        tgt_text_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        '''
        Arguments:
            src_text: (B, T)
            src_text_len: (B,)
            tgt_text: (B, T)
            tgt_text_len: (B,)
        Returns:
            decoder_output: (T, B, C)
        '''
        src_text = self.embedding(src_text) * math.sqrt(512)
        src_text = self.posenc(src_text)
        src_text = src_text.permute(1, 0, 2)    # (T, B, C)
        # tgt_text = self.embedding(tgt_text) * math.sqrt(512)
        # tgt_text = self.posenc(tgt_text)
        # tgt_text = tgt_text.permute(1, 0, 2)    # (T, B, C)
        
        src_padding_mask = torch.arange(src_text.shape[0]).unsqueeze(0).expand(src_text.shape[1], -1).to(device=src_text.device)
        src_padding_mask = src_padding_mask > src_text_len.unsqueeze(-1)
        # tgt_padding_mask = torch.arange(tgt_text.shape[0]).unsqueeze(0).expand(tgt_text.shape[1], -1).to(device=src_text.device)
        # tgt_padding_mask = tgt_padding_mask > tgt_text_len.unsqueeze(-1)
        
        encoder_output = self.encoder(
            src=src_text,
            src_key_padding_mask=src_padding_mask,
        )
        
        text_pred = torch.zeros(1, src_text.shape[1], src_text.shape[2]).to(src_text.device)
        for i in range(iter_limit):
            causal_mask = torch.triu(torch.full((text_pred.shape[0], text_pred.shape[0]), True, device=text_pred.device), diagonal=1)
            decoder_output = self.decoder(
                tgt=text_pred,
                memory=encoder_output,
                tgt_mask=causal_mask,
                # tgt_key_padding_mask=tgt_padding_mask[:, :i + 1],
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_padding_mask,
            )
            decoder_output_future = decoder_output[-1, :, :].unsqueeze(0)
            text_pred = torch.cat((text_pred, decoder_output_future), dim=0)
        text_pred = text_pred[1:, :, :]
        text_pred = self.output_layer(text_pred)
        return text_pred