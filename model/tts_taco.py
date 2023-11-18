from model.taco import PreNet, Attention, ZoneOutCell, TacotronDecoder, ConvEncoder, TacotronDecoderWithQuantizer
from model.pre_post import Postnet

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.vq_taco import *

IGNORE_INDEX = 0

class TTSEncoder(nn.Module):
    def __init__(self, n_vocab, hidden_channels, conv_n_layers, conv_kernel_size, rnn_n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=IGNORE_INDEX)
        conv_layers = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_n_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=conv_kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lstm = nn.LSTM(
            hidden_channels, hidden_channels // 2, num_layers=rnn_n_layers, batch_first=True, bidirectional=True
        )

    def forward(self, x, text_len, spk_emb=None):
        """
        x : (B, T)
        text_len : (B,)
        spk_emb : (B, C)
        """
        x = self.emb(x)     # (B, T, C)
        x = x.permute(0, 2, 1)      # (B, C, T)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)      # (B, T, C)

        seq_len_orig = x.shape[1]
        x = pack_padded_sequence(x, text_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]

        # 複数GPUを使用して最大系列長のデータがバッチ内に含まれない場合などに,系列長が短くなってしまうので再度パディング
        if x.shape[1] < seq_len_orig:
            zero_pad = torch.zeros(x.shape[0], seq_len_orig - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_pad], dim=1)

        return x    # (B, T, C)
    
class TTSTacotron(nn.Module):
    def __init__(self, cfg):
        super().__init__()
      
        self.encoder = TTSEncoder(
            n_vocab=52,
            hidden_channels=256,
            conv_n_layers=3,
            conv_kernel_size=5,
            rnn_n_layers=1,
            dropout=0.5,
        )
        
        self.decoder = TacotronDecoder(
            enc_channels=256,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=cfg.model.post_hidden_channels,
            out_channels=80
        )

    def forward(self, text, text_len, feature_target=None):
        """
        text : (B, len_text)
        text_len : (B,)
        feature_target : (B, 80, T)
        """
        enc_output = self.encoder(text, text_len) #(B, len_text, 512)
        dec_output, logit, att_w, att_c = self.decoder(enc_output, text_len, feature_target, training_method='tf', mode='tts')
        output = self.postnet(dec_output)
        
        return dec_output, output, logit, att_w, att_c
    
class TTSTacotronVq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
      
        self.encoder = TTSEncoder(
            n_vocab=52,
            hidden_channels=256,
            conv_n_layers=3,
            conv_kernel_size=5,
            rnn_n_layers=1,
            dropout=0.5,
        )
        
        self.decoder = TacotronDecoder(
            enc_channels=256,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=cfg.model.post_hidden_channels,
            out_channels=80
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings=2048, embedding_dim=256,
                                        commitment_cost=1.0)
        
    def forward(self, text, text_len, feature_target=None):
        """
        text : (B, len_text)
        text_len : (B,)
        feature_target : (B, 80, T)
        """
        enc_output = self.encoder(text, text_len) #(B, len_text, 512)
        #vq_loss, quantized, perplexity, _ = self._vq_vae(enc_output)
        
        dec_output, logit, att_w, vq_loss, perplexity = self.decoder(enc_output, text_len, feature_target, training_method='tf', mode='tts', vq=self._vq_vae)
        output = self.postnet(dec_output)
        
        return dec_output, output, logit, att_w, vq_loss, perplexity

class TTSTacotronVq128(nn.Module):
    def __init__(self, cfg):
        super().__init__()
      
        self.encoder = TTSEncoder(
            n_vocab=52,
            hidden_channels=128,
            conv_n_layers=3,
            conv_kernel_size=5,
            rnn_n_layers=1,
            dropout=0.5,
        )
        
        self.decoder = TacotronDecoder(
            enc_channels=128,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=cfg.model.post_hidden_channels,
            out_channels=80
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings=2048, embedding_dim=128,
                                        commitment_cost=1.0)
        
    def forward(self, text, text_len, feature_target=None):
        """
        text : (B, len_text)
        text_len : (B,)
        feature_target : (B, 80, T)
        """
        enc_output = self.encoder(text, text_len) #(B, len_text, 512)
        #vq_loss, quantized, perplexity, _ = self._vq_vae(enc_output)

        dec_output, logit, att_w, vq_loss, perplexity = self.decoder(enc_output, text_len, feature_target, training_method='tf', mode='tts', vq=self._vq_vae)
        output = self.postnet(dec_output)
        
        return dec_output, output, logit, att_w, vq_loss, perplexity

class TTSTacotronVq128EMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
      
        self.encoder = TTSEncoder(
            n_vocab=52,
            hidden_channels=128,
            conv_n_layers=3,
            conv_kernel_size=5,
            rnn_n_layers=1,
            dropout=0.5,
        )
        
        self.decoder = TacotronDecoder(
            enc_channels=128,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=cfg.model.post_hidden_channels,
            out_channels=80
        )
        
        self._vq_vae = VectorQuantizerEMA(num_embeddings=2048, embedding_dim=128,
                                        commitment_cost=1.0)
        
    def forward(self, text, text_len, feature_target=None):
        """
        text : (B, len_text)
        text_len : (B,)
        feature_target : (B, 80, T)
        """
        enc_output = self.encoder(text, text_len) #(B, len_text, 512)
        #vq_loss, quantized, perplexity, _ = self._vq_vae(enc_output)

        dec_output, logit, att_w, vq_loss, perplexity = self.decoder(enc_output, text_len, feature_target, training_method='tf', mode='tts', vq=self._vq_vae)
        output = self.postnet(dec_output)
        
        return dec_output, output, logit, att_w, vq_loss, perplexity
    
class TTSTacotronVqRe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
      
        self.encoder = TTSEncoder(
            n_vocab=52,
            hidden_channels=256,
            conv_n_layers=3,
            conv_kernel_size=5,
            rnn_n_layers=1,
            dropout=0.5,
        )
        
        self.decoder = TacotronDecoder(
            enc_channels=256,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=cfg.model.post_hidden_channels,
            out_channels=80
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings=64, embedding_dim=256,
                                        commitment_cost=0.25)
        
    def forward(self, text, text_len, feature_target=None):
        """
        text : (B, len_text)
        text_len : (B,)
        feature_target : (B, 80, T)
        """
        enc_output = self.encoder(text, text_len) #(B, len_text, 512)
        vq_loss, quantized, perplexity, _ = self._vq_vae(enc_output)
        
        dec_output, logit, att_w = self.decoder(quantized, text_len, feature_target, training_method='tf', mode='tts')
        output = self.postnet(dec_output)
        
        return dec_output, output, logit, att_w, vq_loss, perplexity

    
class TTSTacotronQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
      
        self.encoder = TTSEncoder(
            n_vocab=52,
            hidden_channels=256,
            conv_n_layers=3,
            conv_kernel_size=5,
            rnn_n_layers=1,
            dropout=0.5,
        )
        
        self.decoder = TacotronDecoderWithQuantizer(
            enc_channels=256,
            dec_channels=1024,
            atten_conv_channels=32,
            atten_conv_kernel_size=31,
            atten_hidden_channels=128,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=2,
            dropout=0.1,
            use_gc=False
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=cfg.model.post_hidden_channels,
            out_channels=80
        )
        
        self._vq_vae = Quantizer(num_embeddings=2048, embedding_dim=256)
        
    def forward(self, text, text_len, feature_target=None):
        """
        text : (B, len_text)
        text_len : (B,)
        feature_target : (B, 80, T)
        """
        enc_output = self.encoder(text, text_len) #(B, len_text, 512)

        dec_output, logit, att_w, att_c = self.decoder(enc_output, text_len, feature_target, training_method='tf', mode='tts', vq=self._vq_vae)
        output = self.postnet(dec_output)
        
        return dec_output, output, logit, att_w