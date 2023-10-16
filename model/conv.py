"""
gluで推論時にincremental forwardを使うために使用する畳み込み層
wavenet参照
"""

import torch
from packaging import version
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

torch_is_ge_180 = version.parse(torch.__version__) >= version.parse("1.8.0")


class Conv1d(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        if torch_is_ge_180:
            self.register_full_backward_hook(self._clear_linearized_weight)
        else:
            self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        """
        推論時に行われる処理です
        self.input_bufferに前時刻の出力を保持していくことで、前時刻を考慮した推論を実現しています
        あるデータに対しての出力が終わったらclear_bufferメソッドを使用し、bufferを消去します

        input : (B, T=1, C)

        return 
        output : (B, T=1, C)
        """
        # 学習時は行われないようにする
        if self.training:
            raise RuntimeError("incremental_forward only supports eval mode")
        
        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        
        # reshape weight
        weight = self._get_linearized_weight()  
        kw = self.kernel_size[0]
        dilation = self.dilation[0]
        bsz = input.size(0)  # input: bsz x len x dim
        
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                # bufferがない場合、kernel_size、dilationを考慮したbufferを新たに作成
                self.input_buffer = input.new(
                    bsz, kw + (kw - 1) * (dilation - 1), input.size(2)
                )
                # 0初期化
                self.input_buffer.zero_()
            else:
                # 前時刻で生成したbufferがある場合、その値をシフト（前時刻の出力をkernel_size分だけ保持するようになってます）
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            
            # 今回の入力をbufferに代入
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
    
        with torch.no_grad():
            # 畳み込みと同様の計算を全結合層を用いて行っている
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            assert self.weight.size() == (self.out_channels, self.in_channels, kw)
            weight = self.weight.transpose(1, 2).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, apply_weight_norm=False, *args, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        if apply_weight_norm:
            self.conv = weight_norm(Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, *args, **kwargs))
        else:
            self.conv = Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, *args, **kwargs)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, incremental):
        """
        x : (B, C, T)
        return : (B, C, T)
        """
        # 1 次元畳み込み
        if incremental:
            x = x.permute(0, -1, 1)     # (B, T, C)
            y = self.conv.incremental_forward(x)
            y = y.permute(0, -1, 1)     # (B, C, T)
        else:
            y = self.conv(x)
            # 因果性を担保するために、順方向にシフトする
            if self.padding > 0:
                y = y[:, :, :-self.padding]
        return y

    def clear_buffer(self):
        self.conv.clear_buffer()