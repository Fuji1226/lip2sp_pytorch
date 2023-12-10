import torch
from torch import nn
import torch.nn.functional as F

from .net import ResNet3D, ResNet3D_redu2
from .transformer_remake import Encoder
from .pre_post import Postnet

from .vq_vae import ContentEncoder, VectorQuantizer, ResTCDecoder, VectorQuantizerEMA, VectorQuantizerForFineTune, VectorQuantizerForFineTuneWithMLM, ResTCDecoder_Redu4
from .mlm import MLMTrainer

from .vq_vae import VAE_TacotronDecoder

class Lip2Sp_VQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb_dim = 80
        
        self.ResNet_GAP = ResNet3D(
            in_channels=3, 
            out_channels=emb_dim, 
            inner_channels=128,
            layers=3, 
            dropout=0.5,
            norm_type='in',
        )
        
        self.encoder = Encoder(
            n_layers=2, 
            n_head=4, 
            d_model=emb_dim, 
            reduction_factor=2,  
        )

        self.vq = VectorQuantizerForFineTune(num_embeddings=80, embedding_dim=emb_dim, commitment_cost=0.25)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=emb_dim,
            out_channels=80,
            inner_channels=256,
            n_layers=3,
            kernel_size=5,
            dropout=0.5,
            feat_add_channels=80, 
            feat_add_layers=80,
            use_feat_add=False,
            phoneme_classes=53,
            use_phoneme=False,
            n_attn_layer=1,
            n_head=4,
            d_model=emb_dim,
            reduction_factor=1,
            use_attention=False,
            compress_rate=2,
            upsample_method='conv'
        )
        
    def forward(self, lip, data_len, vq_idx=None):
        all_out = {}
        
        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        enc_output = self.encoder(lip_feature, data_len)  
        
        if vq_idx is not None:
            enc_loss = self.calc_metric_enc_output(enc_output, vq_idx, data_len)
            all_out['enc_loss'] = enc_loss
            
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)
        output = self.decoder(vq, data_len)
    
        
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding

        return all_out
    
    def calc_metric_enc_output(self, enc_output, vq_idx, data_len):
        data_len = torch.floor_divide(data_len, 2)
        def create_mask(lengths, enc_output):
            device = lengths.device

            if not isinstance(lengths, list):
                lengths = lengths.tolist()
            bs = int(len(lengths))
            
            max_len = int(max(lengths))
                
            seq_range = torch.arange(0, max_len, dtype=torch.int64)
            seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
            seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
            mask = seq_range_expand < seq_length_expand
            mask = mask.unsqueeze(-1).repeat(1, 1, enc_output.shape[-1])
            
            return mask

        mask = create_mask(data_len, enc_output).to(enc_output.device)
        mask_enc = enc_output.masked_select(mask)
        
        vq_list = []
        
        for i in range(len(data_len)):
            tmp = vq_idx[i, :data_len[i]]
            tmp = self.vq._embedding(tmp).view(-1)
            
            vq_list.append(tmp)
        
        vq_list = torch.cat(vq_list, dim=-1)

        loss = F.mse_loss(mask_enc, vq_list)
        return loss
        
class Lip2Sp_VQVAE_TacoAR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb_dim = 256
        self.reduction_factor = 2
        
        self.ResNet_GAP = ResNet3D(
            in_channels=3, 
            out_channels=emb_dim, 
            inner_channels=128,
            layers=3, 
            dropout=0.5,
            norm_type='in',
        )
        
        self.encoder = Encoder(
            n_layers=2, 
            n_head=4, 
            d_model=emb_dim, 
            reduction_factor=self.reduction_factor,  
        )

        self.vq = VectorQuantizerForFineTune(num_embeddings=512, embedding_dim=emb_dim, commitment_cost=0.25, reduction_factor=self.reduction_factor)

        self.decoder = VAE_TacotronDecoder(
            enc_channels=emb_dim,
            dec_channels=1024,
            rnn_n_layers=2,
            prenet_hidden_channels=256,
            prenet_n_layers=2,
            out_channels=80,
            reduction_factor=self.reduction_factor,
            dropout=0.1,
        )
        
        self.postnet = Postnet(
            in_channels=80,
            inner_channels=512,
            out_channels=80
        )
        self.ctc_output_layer = nn.Linear(emb_dim, 53)
        
        
    def forward(self, lip, data_len, feature=None, mode='inference', reference=None):
        all_out = {}

        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        enc_output = self.encoder(lip_feature, data_len)  
            
        if reference is not None:
            ref_loss = self.calc_ref_loss(enc_output, reference, data_len, enc_output.device)
            all_out['ref_loss'] = ref_loss
            
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)

        if mode != 'inference':
            output, logit = self.decoder(vq, feature)
        else:
            output, logit = self.decoder(vq, None)
    
        ctc_output = self.ctc_output_layer(vq)
        
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding
        all_out['ctc_output'] = ctc_output
        all_out['logit'] = logit

        return all_out
    
    def calc_metric_enc_output(self, enc_output, vq_idx, data_len):
        data_len = torch.floor_divide(data_len, 2)
        def create_mask(lengths, enc_output):
            device = lengths.device

            if not isinstance(lengths, list):
                lengths = lengths.tolist()
            bs = int(len(lengths))
            
            max_len = int(max(lengths))
                
            seq_range = torch.arange(0, max_len, dtype=torch.int64)
            seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
            seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
            mask = seq_range_expand < seq_length_expand
            mask = mask.unsqueeze(-1).repeat(1, 1, enc_output.shape[-1])
            
            return mask

        mask = create_mask(data_len, enc_output).to(enc_output.device)
        mask_enc = enc_output.masked_select(mask)
        
        vq_list = []
        
        for i in range(len(data_len)):
            tmp = vq_idx[i, :data_len[i]]
            tmp = self.vq._embedding(tmp).view(-1)
            
            vq_list.append(tmp)
        
        vq_list = torch.cat(vq_list, dim=-1)

        loss = F.mse_loss(mask_enc, vq_list)
        return loss
    
    def calc_ref_loss(self, enc_output, reference, lengths, device):
        mask = self.create_enc_mask(enc_output, lengths).to(device)

        mask_enc = enc_output.masked_select(mask)
        mask_ref = reference.masked_select(mask)
        
        loss = F.mse_loss(mask_enc, mask_ref)
        return loss
        
    def create_enc_mask(self, enc_output, lengths):
        device = lengths.device
        lengths = torch.div(lengths, self.reduction_factor, rounding_mode='floor')
        
        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        
        max_len = int(max(lengths))
            
        seq_range = torch.arange(0, max_len, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand < seq_length_expand
        mask = mask.unsqueeze(-1).repeat(1, 1, enc_output.shape[-1])
        
        return mask
    
class Lip2Sp_VQVAE_AR_Redu4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb_dim = 80
        
        self.ResNet_GAP = ResNet3D_redu2(
            in_channels=3, 
            out_channels=emb_dim, 
            inner_channels=128,
            layers=3, 
            dropout=0.5,
            norm_type='in',
        )
        
        self.encoder = Encoder(
            n_layers=2, 
            n_head=4, 
            d_model=emb_dim, 
            reduction_factor=4,  
        )

        self.vq = VectorQuantizerForFineTune(num_embeddings=80, embedding_dim=emb_dim, commitment_cost=0.25, reduction_factor=2)

        self.decoder = ResTCDecoder_Redu4(
            cond_channels=emb_dim,
            out_channels=80,
            inner_channels=256,
            n_layers=3,
            kernel_size=5,
            dropout=0.5,
            feat_add_channels=80, 
            feat_add_layers=80,
            use_feat_add=False,
            phoneme_classes=53,
            use_phoneme=False,
            n_attn_layer=1,
            n_head=4,
            d_model=emb_dim,
            reduction_factor=1,
            use_attention=False,
            compress_rate=2,
            upsample_method='conv'
        )
        
        
    def forward(self, lip, data_len, vq_idx=None):
        all_out = {}

        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        enc_output = self.encoder(lip_feature, data_len)  
            
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)
        output = self.decoder(vq, data_len)
    
        
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding

        return all_out
    
    def calc_metric_enc_output(self, enc_output, vq_idx, data_len):
        data_len = torch.floor_divide(data_len, 2)
        def create_mask(lengths, enc_output):
            device = lengths.device

            if not isinstance(lengths, list):
                lengths = lengths.tolist()
            bs = int(len(lengths))
            
            max_len = int(max(lengths))
                
            seq_range = torch.arange(0, max_len, dtype=torch.int64)
            seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
            seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
            mask = seq_range_expand < seq_length_expand
            mask = mask.unsqueeze(-1).repeat(1, 1, enc_output.shape[-1])
            
            return mask

        mask = create_mask(data_len, enc_output).to(enc_output.device)
        mask_enc = enc_output.masked_select(mask)
        
        vq_list = []
        
        for i in range(len(data_len)):
            tmp = vq_idx[i, :data_len[i]]
            tmp = self.vq._embedding(tmp).view(-1)
            
            vq_list.append(tmp)
        
        vq_list = torch.cat(vq_list, dim=-1)

        loss = F.mse_loss(mask_enc, vq_list)
        return loss
    
class Lip_VQENC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb_dim = 256
        
        self.encoder = ContentEncoder(
            in_channels=80,
            out_channels=emb_dim,
            n_attn_layer=2,
            n_head=4,
            reduction_factor=2,
            norm_type='bn',
        )
    
        self.vq = VectorQuantizerForFineTune(num_embeddings=512, embedding_dim=emb_dim, commitment_cost=0.25, reduction_factor=2)
        
    def forward(self, feature, data_len):
        enc_output = self.encoder(feature, data_len)    
        loss, vq, perplexity, encoding = self.vq(enc_output, data_len)
        
        all_out = {}
        all_out['vq'] = vq
        
        return all_out
    
    
class Lip2Sp_VQVAE_MLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        emb_dim = 80
        
        self.ResNet_GAP = ResNet3D(
            in_channels=3, 
            out_channels=emb_dim, 
            inner_channels=128,
            layers=3, 
            dropout=0.5,
            norm_type='in',
        )
        
        self.encoder = Encoder(
            n_layers=2, 
            n_head=4, 
            d_model=emb_dim, 
            reduction_factor=2,  
        )

        self.vq = VectorQuantizerForFineTuneWithMLM(num_embeddings=80, embedding_dim=emb_dim, commitment_cost=0.25)

        # decoder
        self.decoder = ResTCDecoder(
            cond_channels=emb_dim,
            out_channels=80,
            inner_channels=256,
            n_layers=3,
            kernel_size=5,
            dropout=0.5,
            feat_add_channels=80, 
            feat_add_layers=80,
            use_feat_add=False,
            phoneme_classes=53,
            use_phoneme=False,
            n_attn_layer=1,
            n_head=4,
            d_model=256,
            reduction_factor=1,
            use_attention=False,
            compress_rate=2,
            upsample_method='conv'
        )
        
    def forward(self, lip, data_len, vq_idx=None):
        all_out = {}
        
        lip_feature = self.ResNet_GAP(lip) #(B, C, T)
        enc_output = self.encoder(lip_feature, data_len)  
        
        if vq_idx is not None:
            enc_loss = self.calc_metric_enc_output(enc_output, vq_idx, data_len)
            all_out['enc_loss'] = enc_loss
            
        loss, vq, perplexity, encoding,  mlm_loss= self.vq(enc_output, data_len, vq_idx)
        
        
        output = self.decoder(vq, data_len)
    
        
        all_out['output'] = output
        all_out['vq_loss'] = loss
        all_out['perplexity'] = perplexity
        all_out['encoding'] = encoding
        all_out['mlm_loss'] = mlm_loss

        return all_out
    
    def calc_metric_enc_output(self, enc_output, vq_idx, data_len):
        data_len = torch.floor_divide(data_len, 2)
        def create_mask(lengths, enc_output):
            device = lengths.device

            if not isinstance(lengths, list):
                lengths = lengths.tolist()
            bs = int(len(lengths))
            
            max_len = int(max(lengths))
                
            seq_range = torch.arange(0, max_len, dtype=torch.int64)
            seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
            seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
            mask = seq_range_expand < seq_length_expand
            mask = mask.unsqueeze(-1).repeat(1, 1, enc_output.shape[-1])
            
            return mask

        mask = create_mask(data_len, enc_output).to(enc_output.device)
        mask_enc = enc_output.masked_select(mask)
        
        vq_list = []
        
        for i in range(len(data_len)):
            tmp = vq_idx[i, :data_len[i]]
            tmp = self.vq._embedding(tmp).view(-1)
            
            vq_list.append(tmp)
        
        vq_list = torch.cat(vq_list, dim=-1)

        loss = F.mse_loss(mask_enc, vq_list)
        return loss