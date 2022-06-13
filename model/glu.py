"""
Gated Linear Unit
"""

import os
os.environ['PYTHONBREAKPOINT'] = ''

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

try:
    from .transformer import Prenet
    import conv
except:
    from transformer import Prenet
    import conv


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = weight_norm(conv.Conv1d(
            in_channels, out_channels*2, kernel_size, padding=self.padding, **kwargs))

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, incremental):
        """
        ゲート付き活性化関数に通すため、チャンネル数をout_channels*2にしてます
        """
        # 1 次元畳み込み
        if incremental:
            x = x.permute(0, -1, 1)
            y = self.conv.incremental_forward(x)
        else:
            y = self.conv(x)
            # 因果性を担保するために、順方向にシフトする
            if self.padding > 0:
                y = y[:, :, :-self.padding]
        return y

    def clear_buffer(self):
        self.conv.clear_buffer()


class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropput=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropput)
        self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, feature, x):
        return self._forward(feature, x, False)

    def incremental_forward(self, feature, x):
        return self._forward(feature, x, True)

    def _forward(self, feature, x, incremental):
        """
        feature（encoderからの特徴量） : (B, C, T)
        x（音響特徴量） : (B, C, T)
        """
        res = x
        y = self.dropout(x)
        
        if incremental:
            split_dim = 1
            y = self.causal_conv.incremental_forward(y)     # (B, C, T)
        else:
            split_dim = 1
            y = self.causal_conv(y)
        
        y1, y2 = torch.split(y, y.shape[split_dim] // 2, dim=split_dim)

        feature = F.softsign(self.conv(feature))
        # y2 += feature
        y2 = y2 + feature
        out = F.sigmoid(y1) * y2

        out += res
        out *= 0.5 ** 0.5
        return out

    def clear_buffer(self):
        self.causal_conv.clear_buffer()


class GLU(nn.Module):
    def __init__(
        self, inner_channels, out_channels,
        pre_in_channels, pre_inner_channels,
        reduction_factor=2, n_layers=4):

        super().__init__()

        self.prenet = Prenet(pre_in_channels, inner_channels, pre_inner_channels)
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels

        self.glu_stack = nn.ModuleList([
            GLUBlock(inner_channels, inner_channels) for _ in range(n_layers)
        ])

        self.conv_o = weight_norm(nn.Conv1d(
            inner_channels, self.out_channels * self.reduction_factor, kernel_size=1))

    def forward(self, enc_output, target=None, gc=None, training_method=None, num_passes=None, mixing_prob=None):
        """
        reduction_factorにより、
        口唇動画のフレームの、reduction_factor倍のフレームを同時に出力する

        input
        target : (B, C, T)
        enc_output : (B, T, C)

        return
        dec_output : (B, C, T)
        """
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels
        # 前時刻の出力
        self.pre_out = None

        # global conditionの結合
        
        # reshape for reduction factor
        if target is not None:
            target = target.permute(0, -1, -2)
            target = target.reshape(B, -1, D * self.reduction_factor)
            target = target.permute(0, -1, -2)  

        # Prenet
        target = self.prenet(target)    # (B, d_model, T=150)
        self.pre_out = target

        # GLU stack
        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)
        dec_output = target

        # teacher forcing
        if training_method == "tf":
            # decoder layer
            for layer in self.glu_stack:
                dec_output = layer(enc_output, dec_output)
        
        # scheduled sampling
        elif training_method == "ss":
            # decoder layer
            for layer in self.glu_stack:
                dec_output = layer(enc_output, dec_output)

            for k in range(num_passes):
                # decoderからの出力とtargetをmixing_probに従って混合
                mixing_prob = torch.zeros_like(target) + mixing_prob
                judge = torch.bernoulli(mixing_prob)
                judge[:, :, :k] = 1     # t < kの要素は変更しない
                target = torch.where(judge == 1, target, dec_output)
                dec_output = target

            # decoder layer
            for layer in self.glu_stack:
                dec_output = layer(enc_output, dec_output)

        dec_output = self.conv_o(dec_output)
        dec_output = dec_output.reshape(B, D, -1)   
        return dec_output   

    def inference(self, enc_output, prev=None, gc=None):
        """
        reduction_factorにより、
        口唇動画のフレームの、reduction_factor倍のフレームを同時に出力する

        input
        enc_output : (B, T, C)

        return
        dec_output : (B, C, T)
        """
        B = enc_output.shape[0]
        T = enc_output.shape[1]
        D = self.out_channels
        # 前時刻の出力
        self.pre_out = None
        
        # global conditionの結合
        
        # reshape for reduction factor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if prev is not None:
            prev = prev.permute(0, -1, -2)
            prev = prev.reshape(B, -1, D * self.reduction_factor)
            prev = prev.permute(0, -1, -2)
        else:
            go_frame = torch.zeros(B, D * self.reduction_factor, 1)
            prev = go_frame
            prev = prev.to(device)

        max_decoder_time_steps = T
    
        # メインループ
        outs = []
        enc_output = enc_output.permute(0, -1, -2)  # (B, C, T)
        for t in tqdm(range(max_decoder_time_steps)):
            if t > 0:
                # 最初以外は前時刻の出力を入力する
                prev = dec_output[:, :, -1].unsqueeze(-1)
            
            # Prenet
            pre_out = self.prenet(prev)    # (B, d_model, T=1)
            
            # GLU stack
            dec_output = pre_out
            for layer in self.glu_stack:
                # 推論時はincrementa_forwardを適用
                dec_output = layer.incremental_forward(enc_output[:, :, t].unsqueeze(-1), dec_output)

            dec_output = self.conv_o(dec_output)

            # 出力を保持
            outs.append(dec_output.reshape(B, D, -1)[:, :, -2:])
        
        # 各時刻の出力を結合
        outs = torch.cat(outs, dim=2)
        self.clear_buffer()
        return outs

    def clear_buffer(self):
        for layer in self.glu_stack:
            layer.clear_buffer()


def main():
    # 3DResnetからの出力
    batch = 1
    channels = 4
    t = 10
    enc_output = torch.rand(batch, channels, t)
    enc_output = enc_output.permute(0, 2, 1)    # (B, T, C)

    # 音響特徴量
    feature_channels = 80
    feature = torch.rand(batch, feature_channels, t*2)   # (B, C, T)

    # GLU
    inner_channels = channels
    out_channels = feature_channels
    pre_in_channels = feature_channels * 2
    pre_inner_channels = 32
    net = GLU(inner_channels, out_channels, pre_in_channels, pre_inner_channels)
    out = net(enc_output, feature)
    net.eval()
    out = net.inference(enc_output)
    print(out.shape)

if __name__ == "__main__":
    main()