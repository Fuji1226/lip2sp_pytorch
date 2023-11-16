import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = Encoder()
        self.vq = VectorQuantizer(num_embeddings=512, embedding_dim=256, commitment_cost=0.25)
        self.decoder = Decoder()
        
    def forward(self, feature, data_len):
       
        enc, tmp_data_len = self.encoder(feature, data_len)
        loss, vq, _, _ = self.vq(enc)
        
        out = self.decoder(vq, tmp_data_len)
        
        all_out = {}
        all_out['output'] = out
        all_out['vq_loss'] = loss

        return all_out
        


class Encoder(nn.Module):
    def __init__(self, in_channels=80, out_channels=256):
        super().__init__()
        o = out_channels
        in_cs = [in_channels, out_channels, out_channels]
        out_cs = [o, o, o]
        self.stride = [1, 1, 2]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
            ) for in_c, out_c, s in zip(in_cs, out_cs, self.stride)
        ])
        
        self.lstm = nn.LSTM(
            out_channels, out_channels // 2, num_layers=2, batch_first=True, bidirectional=True
        )

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        out : (B, C)
        """
        out = x
        for layer in self.conv_layers:
            out = layer(out)
            
        for stride in self.stride:
            data_len //= stride
      
        x = out.permute(0, 2, 1)      # (B, T, C)

        
        seq_len_orig = x.shape[1]
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]
        
        if x.shape[1] < seq_len_orig:
            zero_pad = torch.zeros(x.shape[0], seq_len_orig - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)
            
        return out, data_len
    
class Decoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        o = 512
        in_cs = [in_channels, in_channels, in_channels]
        out_cs = [o, o, o]

        self.lstm_layers = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(
                        in_channels, in_channels//2, num_layers=2, batch_first=True, bidirectional=True
                        )
            )
        ])
        
        self.feat_out_layer = nn.Linear(in_channels, int(80 * 2), bias=False)

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        out : (B, C)
        """

        x = x.permute(0, 2, 1)
        seq_len_orig = x.shape[1]
        for layer in self.lstm_layers:
            x = pack_padded_sequence(x, data_len.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = layer(x)
            x = pad_packed_sequence(x, batch_first=True)[0]
    
        x = self.feat_out_layer(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 80, -1)
       
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
    
        """_summary_
        text_lenによる長さを考慮した損失計算

        Args:
            quantized (_type_): _description_
            inputs (_type_): _description_
            inputs_len (_type_): _description_
        """

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings