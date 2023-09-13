import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def make_pad_mask(lengths, max_len):
    """
    口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
    マスクする場所をTrue
    """
    # この後の処理でリストになるので先にdeviceを取得しておく
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand     
    return mask.unsqueeze(1).to(device=device)  # (B, 1, T)


class PrenetForLip2SP(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=128, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(inner_channels, inner_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(inner_channels, out_channels, 1),
            nn.ReLU()
        )

        # self.conv1 = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        # self.conv2 = nn.Conv1d(inner_channels, inner_channels, kernel_size=1)
        # self.conv3 = nn.Conv1d(inner_channels, out_channels, kernel_size=1)

        self.project_pre = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.layer_norm = nn.LayerNorm(out_channels)

        self.dropout = dropout
        

    def forward(self, x):
        """
        音響特徴量をtransformer内の次元に調整する役割
        x : (B, C=feature channels, T)
        y : (B, C=d_model, T)
        """
        # out = F.relu(self.conv1(x))
        # out = F.dropout(out, self.dropout, training=True)

        # out = F.relu(self.conv2(out))
        # out = F.dropout(out, self.dropout, training=True)

        # out = F.relu(self.conv3(out))
        x = x.transpose(-1, -2)
        out = self.fc(x)

        out = self.project_pre(out)
        out = self.layer_norm(out.transpose(-1, -2))

        #breakpoint()
        return out

class PreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels if i == 0 else hidden_channels, hidden_channels),
                    nn.ReLU(),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C)
        """
        # dropoutは推論時も適用(同じ音素が連続するのを防ぐため,前時刻の出力にあえてランダム性を付加する)
        for layer in self.layers:
            x = F.dropout(layer(x))
        return x


class ConvEncoder(nn.Module):
    def __init__(self, hidden_channels, conv_n_layers, conv_kernel_size, rnn_n_layers, dropout):
        super().__init__()
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

    def forward(self, x, data_len):
        """
        x : (B, C, T)
        data_len : (B,)
        """
        tmp_len = data_len.clone().detach()
        # print(f'x: {x.shape}')
        # print(f'data_len: {data_len}')
        #x = x.permute(0, 2, 1)      # (B, C, T)
        for layer in self.conv_layers:
            x = layer(x)
        #print(f'after layer: {x.shape}')
        x = x.permute(0, 2, 1)      # (B, T, C)

        seq_len_orig = x.shape[1]
        
        for i in range(x.shape[0]):
            max_len = x.shape[1]
            
            if tmp_len[i]>max_len:
                tmp_len[i] = max_len
            
        x = pack_padded_sequence(x, tmp_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        #print(f'after lstm: {x}')
        x = pad_packed_sequence(x, batch_first=True)[0]
        
        # # 複数GPUを使用して最大系列長のデータがバッチ内に含まれない場合などに,系列長が短くなってしまうので再度パディング
        # if x.shape[1] < seq_len_orig:
        #     zero_pad = torch.zeros(x.shape[0], seq_len_orig - x.shape[1], x.shape[2]).to(device=x.device, dtype=x.dtype)
        #     x = torch.cat([x, zero_pad], dim=1)

        return x    # (B, T, C)

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

    def forward(self, enc_output, text_len, dec_state, prev_att_w, mask=None):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        dec_state : (B, C)
        prev_att_w : (B, T)
        """
        if self.processed_memory is None:
            self.processed_memory = self.fc_enc(enc_output)     # (B, T, C)

        test = text_len.clone().detach()
        for i in range(test.shape[-1]):
            if test[i] > enc_output.shape[1]:
                test[i] = enc_output.shape[1] 
   

        # if prev_att_w is None:
        #     prev_att_w = 1.0 - make_pad_mask(test, enc_output.shape[1]).squeeze(1).to(torch.float32)   # (B, T)
        #     prev_att_w = prev_att_w / test.unsqueeze(1)
            
        if prev_att_w is None:
            prev_att_w = torch.zeros(enc_output.shape[0], enc_output.shape[1]).to(enc_output.device)
            prev_att_w[:, 0] = 1.0

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
        self, enc_channels, dec_channels, atten_conv_channels, atten_conv_kernel_size, atten_hidden_channels,
        rnn_n_layers, prenet_hidden_channels, prenet_n_layers, out_channels, reduction_factor, dropout, use_gc):
        super().__init__()
        self.enc_channels = enc_channels
        self.prenet_hidden_channels = prenet_hidden_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        self.reduction_factor = reduction_factor

        self.attention = Attention(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conv_channels=atten_conv_channels,
            conv_kernel_size=atten_conv_kernel_size,
            hidden_channels=atten_hidden_channels,
        )

        self.prenet = PreNet(int(out_channels * reduction_factor), prenet_hidden_channels, prenet_n_layers)
        #self.prenet = PrenetForLip2SP(int(out_channels * reduction_factor), prenet_hidden_channels)


        lstm = []
        for i in range(rnn_n_layers):
            lstm.append(
                ZoneOutCell(
                    nn.LSTMCell(
                        enc_channels + prenet_hidden_channels if i == 0 else dec_channels,
                        dec_channels,
                    ), 
                    zoneout=dropout
                )
            )
        self.lstm = nn.ModuleList(lstm)
        self.feat_out_layer = nn.Linear(enc_channels + dec_channels, int(out_channels * reduction_factor), bias=False)
        self.prob_out_layer = nn.Linear(enc_channels + dec_channels, reduction_factor)
        

    def _zero_state(self, hs, i):
        init_hs = hs.new_zeros(hs.size(0), self.dec_channels)
        return init_hs

    def forward(self, enc_output, text_len=None, feature_target=None, training_method=None, mixing_prob=None, use_stop_token=False):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        #print(f'text len: {text_len}')
        if feature_target is not None:
            B, C, T = feature_target.shape
            feature_target = feature_target.permute(0, 2, 1)
            feature_target = feature_target.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))
        else:
            B = enc_output.shape[0]
            C = self.out_channels
            
        if feature_target is not None:
            max_decoder_time_step = feature_target.shape[1]
        else:
            max_decoder_time_step = enc_output.shape[1]

        mask = make_pad_mask(text_len, enc_output.shape[1]).squeeze(1)      # (B, T)

        h_list, c_list = [], []
        for i in range(len(self.lstm)):
            h_list.append(self._zero_state(enc_output, i))
            c_list.append(self._zero_state(enc_output, i))

        go_frame = enc_output.new_zeros(enc_output.size(0), int(self.out_channels * self.reduction_factor))
        prev_out = go_frame

        prev_att_w = None
        self.attention.reset()

        output_list = []
        logit_list = []
        att_w_list = []
        t = 0

        # if feature_target is not None:
        #     print(f'featue target: {feature_target.shape}')
        #     print(f'training method: {training_method}')
        while True:
            att_c, att_w = self.attention(enc_output, text_len, h_list[0], prev_att_w, mask=mask)
            #enc_idx = int(t//2)
            #att_c = enc_output[:, enc_idx, :]

            prenet_out = self.prenet(prev_out)      # (B, C)

            xs = torch.cat([att_c, prenet_out], dim=1)      # (B, C)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            
            hcs = torch.cat([h_list[-1], att_c], dim=1)     # (B, C)
            output = self.feat_out_layer(hcs)   # (B, C)
            logit = self.prob_out_layer(hcs)    # (B, reduction_factor)
            output_list.append(output)
            logit_list.append(logit)
            att_w_list.append(att_w)

            # if feature_target is not None:
            #     prev_out = feature_target[:, t, :]
            # else:
            #     prev_out = output
            if feature_target is not None:
                if training_method == "tf":
                    prev_out = feature_target[:, t, :]
                    
                elif training_method == "ss":
                    """
                    mixing_prob = 1 : teacher forcing
                    mixing_prob = 0 : using decoder prediction completely
                    """
                    judge = torch.bernoulli(torch.tensor(mixing_prob))
                    if judge:
                        prev_out = feature_target[:, t, :]
                    else:
                        #prev_out = output.clone().detach()
                        prev_out = output.clone().detach()
            else:
                prev_out = output

            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
    
            if t > max_decoder_time_step - 1:
                break
            if feature_target is None and (torch.sigmoid(logit) >= 0.5).any():
                break
        
        output = torch.cat(output_list, dim=1)  # (B, T, C)
        output = output.reshape(B, -1, C).permute(0, 2, 1)  # (B, C, T)
        logit = torch.cat(logit_list, dim=-1)   # (B, T)
        
        att_w = torch.stack(att_w_list, dim=1)  # (B, T, C)
        
        if not use_stop_token:
            return output, logit, att_w #(B, mel, T) (B, T)
        else:
            return output, logit, att_w, logit_list



class TacotronDecoder(nn.Module):
    def __init__(
        self, enc_channels, dec_channels, atten_conv_channels, atten_conv_kernel_size, atten_hidden_channels,
        rnn_n_layers, prenet_hidden_channels, prenet_n_layers, out_channels, reduction_factor, dropout, use_gc):
        super().__init__()
        self.enc_channels = enc_channels
        self.prenet_hidden_channels = prenet_hidden_channels
        self.dec_channels = dec_channels
        self.out_channels = out_channels
        self.reduction_factor = reduction_factor

        self.attention = Attention(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conv_channels=atten_conv_channels,
            conv_kernel_size=atten_conv_kernel_size,
            hidden_channels=atten_hidden_channels,
        )

        self.prenet = PreNet(int(out_channels * reduction_factor), prenet_hidden_channels, prenet_n_layers)
        #self.prenet = PrenetForLip2SP(int(out_channels * reduction_factor), prenet_hidden_channels)


        lstm = []
        for i in range(rnn_n_layers):
            lstm.append(
                ZoneOutCell(
                    nn.LSTMCell(
                        enc_channels + prenet_hidden_channels if i == 0 else dec_channels,
                        dec_channels,
                    ), 
                    zoneout=dropout
                )
            )
        self.lstm = nn.ModuleList(lstm)
        self.feat_out_layer = nn.Linear(enc_channels + dec_channels, int(out_channels * reduction_factor), bias=False)
        self.prob_out_layer = nn.Linear(enc_channels + dec_channels, reduction_factor)
        

    def _zero_state(self, hs, i):
        init_hs = hs.new_zeros(hs.size(0), self.dec_channels)
        return init_hs

    def forward(self, enc_output, text_len=None, feature_target=None, training_method=None, mixing_prob=None, use_stop_token=False):
        """
        enc_output : (B, T, C)
        text_len : (B,)
        feature_target : (B, C, T)
        spk_emb : (B, C)
        """
        #print(f'text len: {text_len}')
        if feature_target is not None:
            B, C, T = feature_target.shape
            feature_target = feature_target.permute(0, 2, 1)
            feature_target = feature_target.reshape(B, T // self.reduction_factor, int(C * self.reduction_factor))
        else:
            B = enc_output.shape[0]
            C = self.out_channels
            
        if feature_target is not None:
            max_decoder_time_step = feature_target.shape[1]
        else:
            max_decoder_time_step = enc_output.shape[1]  * int(2//self.reduction_factor)

        mask = make_pad_mask(text_len, enc_output.shape[1]).squeeze(1)      # (B, T)

        h_list, c_list = [], []
        for i in range(len(self.lstm)):
            h_list.append(self._zero_state(enc_output, i))
            c_list.append(self._zero_state(enc_output, i))

        go_frame = enc_output.new_zeros(enc_output.size(0), int(self.out_channels * self.reduction_factor))
        prev_out = go_frame

        prev_att_w = None
        self.attention.reset()

        output_list = []
        logit_list = []
        att_w_list = []
        t = 0

        # if feature_target is not None:
        #     print(f'featue target: {feature_target.shape}')
        #     print(f'training method: {training_method}')
        while True:
            att_c, att_w = self.attention(enc_output, text_len, h_list[0], prev_att_w, mask=mask)
            #test attentionベクトルを固定
            # enc_index = int(t//2)
            # att_c = enc_output[:, enc_index, :]
            
            
            prenet_out = self.prenet(prev_out)      # (B, C)

            xs = torch.cat([att_c, prenet_out], dim=1)      # (B, C)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            
            hcs = torch.cat([h_list[-1], att_c], dim=1)     # (B, C)
            output = self.feat_out_layer(hcs)   # (B, C)
            logit = self.prob_out_layer(hcs)    # (B, reduction_factor)
            output_list.append(output)
            logit_list.append(logit)
            att_w_list.append(att_w)

            # if feature_target is not None:
            #     prev_out = feature_target[:, t, :]
            # else:
            #     prev_out = output
            if feature_target is not None:
                if training_method == "tf":
                    prev_out = feature_target[:, t, :]
                    
                elif training_method == "ss":
                    """
                    mixing_prob = 1 : teacher forcing
                    mixing_prob = 0 : using decoder prediction completely
                    """
                    judge = torch.bernoulli(torch.tensor(mixing_prob))
                    if judge:
                        prev_out = feature_target[:, t, :]
                    else:
                        prev_out = output.clone().detach()
                        #prev_out = output
            else:
                prev_out = output

            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
    
            if t > max_decoder_time_step - 1:
                break
        
        output = torch.cat(output_list, dim=1)  # (B, T, C)
        output = output.reshape(B, -1, C).permute(0, 2, 1)  # (B, C, T)
        logit = torch.cat(logit_list, dim=-1)   # (B, T)
        
        att_w = torch.stack(att_w_list, dim=1)  # (B, T, C)
        
        if not use_stop_token:
            return output, logit, att_w #(B, mel, T) (B, T)
        else:
            return output, logit, att_w, logit
