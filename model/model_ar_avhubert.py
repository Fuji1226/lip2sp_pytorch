import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from avhubert import MyAVHubertModel
from glu_remake import GLU
from model.transformer_remake import make_pad_mask


def load_avhubert_torch(
    cfg,
    model_size,
    load_pretrained_weight,
    layer_loaded,
):
    if model_size == 'base':
        avhubert = MyAVHubertModel(cfg.base)
        ckpt_path = Path(cfg.base.ckpt_path).expanduser()
    elif model_size == 'large':
        avhubert = MyAVHubertModel(cfg.large)
        ckpt_path = Path(cfg.large.ckpt_path).expanduser()

    if load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))[str('avhubert')]
        avhubert_dict = avhubert.state_dict()

        # 読み込むパラメータを選択
        if layer_loaded == 'transformer':
            required_state_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('encoder.')}
        elif layer_loaded == 'resnet':
            required_state_dict = {k: v for k, v in pretrained_dict.items() if not (k.startswith('encoder.'))}
        elif layer_loaded == 'all':
            required_state_dict = pretrained_dict

        # state_dictは辞書なので、読み込みたいkey（layer）だけ重みをupdate
        avhubert_dict.update(required_state_dict)
        avhubert.load_state_dict(avhubert_dict)

    return avhubert


class Attention(nn.Module):
    def __init__(self, enc_channels, dec_channels, conv_channels, conv_kernel_size, hidden_channels):
        super().__init__()
        self.fc_enc = nn.Linear(enc_channels, hidden_channels)
        self.fc_dec = nn.Linear(dec_channels, hidden_channels, bias=False)
        self.fc_att = nn.Linear(conv_channels, hidden_channels, bias=False)
        self.loc_conv = nn.Conv1d(1, conv_channels, conv_kernel_size, padding=(conv_kernel_size - 1) // 2, bias=False)
        self.w = nn.Linear(hidden_channels, 1)
        self.processed_memory = None

    def reset(self):
        self.processed_memory = None

    def forward(self, enc_output, lip_len, dec_state, prev_att_w, mask=None):
        """
        enc_output : (B, T, C)
        lip_len : (B,)
        dec_state : (B, C)
        prev_att_w : (B, T)
        """
        if self.processed_memory is None:
            self.processed_memory = self.fc_enc(enc_output)     # (B, T, C)

        if prev_att_w is None:
            prev_att_w = 1.0 - make_pad_mask(lip_len, enc_output.shape[1]).squeeze(1).to(torch.float32)   # (B, T)
            prev_att_w = prev_att_w / lip_len.unsqueeze(1)

        att_conv = self.loc_conv(prev_att_w.unsqueeze(1))     # (B, C, T)
        att_conv = att_conv.permute(0, 2, 1)    # (B, T, C)
        att_conv = self.fc_att(att_conv)    # (B, T, C)

        dec_state = self.fc_dec(dec_state).unsqueeze(1)      # (B, 1, C)
        
        att_energy = self.w(torch.tanh(att_conv + self.processed_memory + dec_state))   # (B, T, 1)
        att_energy = att_energy.squeeze(-1)     # (B, T)

        if mask is not None:
            att_energy = att_energy.masked_fill(mask, torch.tensor(float('-inf')))

        att_w = F.softmax(att_energy, dim=1)    # (B, T)
        att_c = torch.sum(enc_output * att_w.unsqueeze(-1), dim=1)  # (B, C)
        return att_c, att_w


class PreNet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, inner_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(inner_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (B, C)
        """
        return self.layers(x)
    

class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class TacotronDecoder(nn.Module):
    def __init__(
        self,
        enc_channels,
        dec_channels,
        atten_conv_channels,
        atten_conv_kernel_size,
        atten_hidden_channels,
        out_channels,
        reduction_factor,
        prenet_hidden_channels,
        prenet_inner_channels,
        lstm_n_layers,
        dropout,
    ):
        super().__init__()
        self.enc_channels = enc_channels
        self.prenet_hidden_channels = prenet_hidden_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        self.reduction_factor = reduction_factor
        self.training_method = 'teacher_forcing'
        self.scheduled_sampling_thres = 0

        self.attention = Attention(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conv_channels=atten_conv_channels,
            conv_kernel_size=atten_conv_kernel_size,
            hidden_channels=atten_hidden_channels,
        )

        self.prenet = PreNet(
            in_channels=out_channels * reduction_factor,
            out_channels=prenet_hidden_channels,
            inner_channels=prenet_inner_channels,
        )

        lstm = []
        for i in range(lstm_n_layers):
            lstm.append(
                ZoneOutCell(
                    nn.LSTMCell(
                        enc_channels + prenet_hidden_channels if i == 0 else dec_channels,
                        dec_channels,
                    ), 
                    zoneout=dropout,
                )
            )
        self.lstm = nn.ModuleList(lstm)
        self.feat_out_layer = nn.Linear(enc_channels + dec_channels, int(out_channels * reduction_factor), bias=False)

    def _zero_state(self, hs, i):
        init_hs = hs.new_zeros(hs.size(0), self.dec_channels)
        return init_hs

    def forward(self, enc_output, lip_len, feature_target=None):
        '''
        enc_output : (B, T, C)
        lip_len : (B,)
        feature_target : (B, C, T)
        '''
        if feature_target is not None:
            B, C, T = feature_target.shape
            feature_target = feature_target.permute(0, 2, 1)    # (B, T, C)
            feature_target = feature_target.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))
        else:
            B = enc_output.shape[0]
            C = self.out_channels

        mask = make_pad_mask(lip_len, enc_output.shape[1]).squeeze(1)      # (B, T)
        
        h_list, c_list = [], []
        for i in range(len(self.lstm)):
            h_list.append(self._zero_state(enc_output, i))
            c_list.append(self._zero_state(enc_output, i))

        go_frame = enc_output.new_zeros(enc_output.size(0), int(self.out_channels * self.reduction_factor))
        prev_out = go_frame

        prev_att_w = None
        self.attention.reset()

        output_list = []
        att_w_list = []

        for t in range(enc_output.shape[1]):
            # att_c, att_w = self.attention(enc_output, lip_len, h_list[0], prev_att_w, mask=mask)
            att_c = enc_output[:, t, :]
            att_w = torch.rand(B, 250)
            prenet_out = self.prenet(prev_out)      # (B, C)

            xs = torch.cat([att_c, prenet_out], dim=1)      # (B, C)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            
            hcs = torch.cat([h_list[-1], att_c], dim=1)     # (B, C)
            output = self.feat_out_layer(hcs)   # (B, C)

            output_list.append(output)
            att_w_list.append(att_w)

            if feature_target is not None:
                if self.training_method == 'teacher_forcing':
                    prev_out = feature_target[:, t, :]
                elif self.training_method == 'scheduled_sampling':
                    x = torch.randint(0, 100, (1,))
                    if x < self.scheduled_sampling_thres:
                        prev_out = output
                    else:
                        prev_out = feature_target[:, t, :]
            else:
                prev_out = output

            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

        output = torch.cat(output_list, dim=1)  # (B, T, C)
        output = output.reshape(B, -1, C).permute(0, 2, 1)  # (B, C, T)
        att_w = torch.stack(att_w_list, dim=1)  # (B, T, T)
        return output, att_w
    
    def reset_state(self):
        pass
    

class PostNet(nn.Module):
    def __init__(self, out_channels, hidden_channels, n_layers, kernel_size, dropout=0.5):
        super().__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        for i in range(n_layers - 1):
            in_channels = out_channels if i == 0 else hidden_channels
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Lip2SP_AR_AVHubert(nn.Module):
    def __init__(
        self,
        avhubert_config,
        avhubert_model_size,
        avhubert_return_res_output,
        load_avhubert_pretrained_weight,
        avhubert_layer_loaded,
        reduction_factor,
        which_decoder,
        out_channels,
        dec_dropout,
        use_spk_emb,
        spk_emb_dim,
        pre_inner_channels,
        glu_layers,
        glu_kernel_size,
        dec_channels,
        dec_atten_conv_channels,
        dec_atten_conv_kernel_size,
        dec_atten_hidden_channels,
        prenet_hidden_channels,
        prenet_inner_channels,
        lstm_n_layers,
        post_inner_channels,
        post_n_layers,
        post_kernel_size,
    ):
        super().__init__()
        self.avhubert_return_res_output = avhubert_return_res_output
        self.which_decoder = which_decoder
        self.avhubert = load_avhubert_torch(
            cfg=avhubert_config,
            model_size=avhubert_model_size,
            load_pretrained_weight=load_avhubert_pretrained_weight,
            layer_loaded=avhubert_layer_loaded,
        )
        inner_channels = self.avhubert.encoder_embed_dim

        if use_spk_emb:
            self.spk_emb_layer = nn.Conv1d(inner_channels + spk_emb_dim, inner_channels, kernel_size=1)

        if which_decoder == "glu":
            self.decoder = GLU(
                inner_channels=inner_channels, 
                out_channels=out_channels,
                pre_in_channels=int(out_channels * reduction_factor), 
                pre_inner_channels=pre_inner_channels,
                cond_channels=inner_channels,
                reduction_factor=reduction_factor, 
                n_layers=glu_layers,
                kernel_size=glu_kernel_size,
                dropout=dec_dropout,
                use_spk_emb=use_spk_emb,
                spk_emb_dim=spk_emb_dim,
            )
        elif which_decoder == 'tacotron':
            self.decoder = TacotronDecoder(
                enc_channels=inner_channels,
                dec_channels=dec_channels,
                atten_conv_channels=dec_atten_conv_channels,
                atten_conv_kernel_size=dec_atten_conv_kernel_size,
                atten_hidden_channels=dec_atten_hidden_channels,
                out_channels=out_channels,
                reduction_factor=reduction_factor,
                prenet_hidden_channels=prenet_hidden_channels,
                prenet_inner_channels=prenet_inner_channels,
                lstm_n_layers=lstm_n_layers,
                dropout=dec_dropout,
            )

        self.postnet = PostNet(
            out_channels=out_channels,
            hidden_channels=post_inner_channels,
            n_layers=post_n_layers,
            kernel_size=post_kernel_size,
        )

    def forward(
        self,
        lip,
        audio,
        lip_len,
        spk_emb,
        feature_target=None,
    ):
        '''
        lip : (B, C, H, W, T)
        audio : (B, C, T)
        lip_len : (B,)
        spk_emb : (B, C)
        '''
        self.reset_state()

        if lip is not None:
            lip = lip.permute(0, 1, 4, 2, 3)    # (B, C, T, H, W)
            padding_mask_avhubert = torch.zeros(lip.shape[0], lip.shape[2]).to(device=lip.device, dtype=torch.bool)     # (B, T)
        elif audio is not None:
            padding_mask_avhubert = torch.zeros(audio.shape[0], audio.shape[2]).to(device=audio.device, dtype=torch.bool)     # (B, T)

        # padding部分がTrue
        for i, l in enumerate(lip_len):
            padding_mask_avhubert[i, l:] = True
        
        x = self.avhubert(
            video=lip,
            audio=audio, 
            return_res_output=self.avhubert_return_res_output,
            padding_mask=padding_mask_avhubert, 
        )   # (B, T, C)

        if hasattr(self, 'spk_emb_layer'):
            x = x.permute(0, 2, 1)
            spk_emb = spk_emb.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])
            x = torch.cat([x, spk_emb], dim=1)
            x = self.spk_emb_layer(x)
            x = x.permute(0, 2, 1)

        if feature_target is not None:
            dec_output, att_w = self.decoder_forward(x, lip_len, feature_target)
        else:
            dec_output, att_w = self.decoder_inference(x, lip_len)

        output = self.postnet(dec_output)
        return output, dec_output, att_w
    
    def decoder_forward(self, enc_output, lip_len, feature_target, mode='training'):
        '''
        enc_output : (B, T, C)
        '''
        if self.which_decoder == 'glu':
            dec_output = self.decoder(enc_output, mode=mode, prev=feature_target)
            att_w = None
        elif self.which_decoder == 'tacotron':
            dec_output, att_w = self.decoder(enc_output, lip_len, feature_target=feature_target)
        return dec_output, att_w
    
    def decoder_inference(self, enc_output, lip_len, mode='inference'):
        '''
        enc_output : (B, T, C)
        '''
        dec_output_list = []
        decoder_time_steps = enc_output.shape[1]

        if self.which_decoder == 'glu':
            for t in range(decoder_time_steps):
                if t == 0:
                    dec_output = self.decoder(enc_output[:, t, :].unsqueeze(1), mode=mode, prev=None)
                else:
                    dec_output = self.decoder(enc_output[:, t, :].unsqueeze(1), mode=mode, prev=dec_output_list[-1])
                dec_output_list.append(dec_output)
            dec_output = torch.cat(dec_output_list, dim=-1)     # (B, C, T)
            att_w = None
        elif self.which_decoder == 'tacotron':
            dec_output, att_w = self.decoder(enc_output, lip_len, feature_target=None)

        return dec_output, att_w
    
    def reset_state(self):
        self.decoder.reset_state()


# import hydra
# @hydra.main(version_base=None, config_name="config", config_path="../conf")
# def main(cfg):
#     model = Lip2SP_AR_AVHubert(
#         avhubert_config=cfg.model.avhubert_config,
#         avhubert_model_size=cfg.model.avhubert_model_size,
#         avhubert_return_res_output=cfg.model.avhubert_return_res_output,
#         load_avhubert_pretrained_weight=cfg.model.load_avhubert_pretrained_weight,
#         avhubert_layer_loaded=cfg.model.avhubert_layer_loaded,
#         reduction_factor=cfg.model.reduction_factor,
#         which_decoder=cfg.model.which_decoder,
#         out_channels=cfg.model.out_channels,
#         dec_dropout=cfg.train.dec_dropout,
#         use_spk_emb=cfg.train.use_spk_emb,
#         spk_emb_dim=cfg.model.spk_emb_dim,
#         pre_inner_channels=cfg.model.pre_inner_channels,
#         glu_layers=cfg.model.glu_layers,
#         glu_kernel_size=cfg.model.glu_kernel_size,
#         dec_channels=cfg.model.taco_dec_channels,
#         dec_atten_conv_channels=cfg.model.taco_dec_conv_channels,
#         dec_atten_conv_kernel_size=cfg.model.taco_dec_conv_kernel_size,
#         dec_atten_hidden_channels=cfg.model.taco_dec_atten_hidden_channels,
#         prenet_hidden_channels=cfg.model.taco_dec_prenet_hidden_channels,
#         prenet_inner_channels=cfg.model.taco_dec_prenet_inner_channels,
#         lstm_n_layers=cfg.model.taco_dec_n_layers,
#         post_inner_channels=cfg.model.post_inner_channels,
#         post_n_layers=cfg.model.post_n_layers,
#         post_kernel_size=cfg.model.post_kernel_size,
#     )
#     model.decoder.training_method = 'teacher_forcing'
#     model.decoder.scheduled_sampling_thres = 0
#     lip = torch.rand(1, 1, 88, 88, 250)
#     feature = torch.rand(lip.shape[0], 80, lip.shape[-1] * cfg.model.reduction_factor)
#     lip_len = torch.randint(100, 250, (lip.shape[0],))
#     spk_emb = torch.rand(lip.shape[0], 256)
#     model.train()
#     output, dec_output, att_w = model(
#         lip=lip,
#         audio=None,
#         lip_len=lip_len,
#         spk_emb=spk_emb,
#         feature_target=feature,
#     )
#     print(output.shape, dec_output.shape, att_w.shape) 

#     model.eval()
#     output, dec_output, att_w = model(
#         lip=lip,
#         audio=None,
#         lip_len=lip_len,
#         spk_emb=spk_emb,
#         feature_target=None,
#     )
#     print(output.shape, dec_output.shape, att_w.shape) 


# if __name__ == '__main__':
#     main()