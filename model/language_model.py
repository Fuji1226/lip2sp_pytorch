from pathlib import Path
import sys
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data_process.phoneme_encode import SOS_INDEX, EOS_INDEX, IGNORE_INDEX
from model.transformer_remake import token_mask, get_subsequent_mask, MultiHeadAttention, PositionwiseFeedForward, posenc


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, mask=None):
        """
        dec_input : (B, T, C)
        """
        dec_output = self.self_attention(dec_input, dec_input, dec_input, mask=mask)
        dec_output = self.fc(dec_output)
        return dec_output


class TransformerLM(nn.Module):
    def __init__(self, n_vocab, d_model, n_head, n_layers):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d_model, padding_idx=IGNORE_INDEX)

        layers = []
        for i in range(n_layers):
            layers.append(DecoderLayer(
                d_model=d_model,
                d_inner=int(d_model * 4),
                n_head=n_head,
                d_k=d_model // n_head,
                d_v=d_model // n_head,
            ))
        self.layers = nn.ModuleList(layers)

        self.out_layer = nn.Linear(d_model, n_vocab)

    def forward(self, x, text_len):
        """
        x : (B, T)
        """
        mask = token_mask(x).unsqueeze(1) | get_subsequent_mask(x)
        output = self.emb(x)    # (B, T, C)
        output = output.permute(0, 2, 1)    # (B, C, T)
        output = output + posenc(output, device=output.device, start_index=0)   # (B, T, T)
        output = output.permute(0, 2, 1)    # (B, T, C)
        for layer in self.layers:
            output = layer(output, mask)
        output = self.out_layer(output).permute(0, 2, 1)    # (B, C, T)
        return output


class RNNLM(nn.Module):
    def __init__(self, n_vocab, hidden_channels, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=IGNORE_INDEX)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=n_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_channels, n_vocab)

    def forward(self, x, text_len):
        """
        text : (B, T)
        text_len : (B,)
        """
        x = self.emb(x)     # (B, T, C)
        x = pack_padded_sequence(x, text_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]
        x = self.fc(x).permute(0, 2, 1)     # (B, C, T)
        return x

    def generate(self, x, n_max_loop=1000):
        """
        x : (B, T)
        """
        output_list = []
        for i in range(x.shape[1]):
            output_list.append(x[:, i].unsqueeze(1))

        for i in range(n_max_loop):
            prev = torch.cat(output, dim=1)
            output = self.emb(prev)
            output, _ = self.lstm(output)
            output = self.fc(output)
            output = torch.argmax(output[:, :, :-1], dim=1)
            output_list.append(output)

            if output == EOS_INDEX:
                break
        
        output = torch.cat(output_list, dim=1)
        return output


if __name__ == "__main__":
    model = TransformerLM(
        n_vocab=44,
        d_model=256,
        n_head=4,
        n_layers=2,
    )
    B = 1
    T = 10
    text = torch.randint(0, 44, (B, T))
    text[:, -5:] = 0
    output = model(text)

    model = RNNLM(
        n_vocab=44,
        hidden_channels=128,
        n_layers=2
    )
    text = torch.randint(0, 44, (1, 20))
    text_len = torch.tensor([text.shape[1]])
    output = model(text, text_len)
    breakpoint()