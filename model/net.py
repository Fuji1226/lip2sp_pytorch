import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F

from cbam import SpatialAttention, ChannnelAttention


class NormLayer3D(nn.Module):
    def __init__(self, in_channels, norm_type):
        super().__init__()
        self.norm_type = norm_type
        self.b_n = nn.BatchNorm3d(in_channels)
        self.i_n = nn.InstanceNorm3d(in_channels)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        if self.norm_type == "bn":
            out = self.b_n(x)
        elif self.norm_type == "in":
            out = self.i_n(x)
        return out


class InvResLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, up_scale=6, sq_r=16, kernel_size=None, pooling=True, c_attn=True, s_attn=True):
        super().__init__()
        self.hidden_channels = int(in_channels * up_scale)
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(self.hidden_channels, norm_type),
            nn.ReLU(),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), groups=self.hidden_channels),
            NormLayer3D(self.hidden_channels, norm_type),
        )
        if pooling:
            self.pool_layer = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        if c_attn:
            self.c_attention = ChannnelAttention(self.hidden_channels, sq_r)

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv3d(self.hidden_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )

        if s_attn:
            self.s_attention = SpatialAttention(kernel_size)

        if in_channels != out_channels:
            self.adjust_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                NormLayer3D(out_channels, norm_type),
            )

    def forward(self, x):
        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)
        
        if hasattr(self, "pool_layer"):
            out = self.pool_layer(out)
            x = self.pool_layer(x)

        if hasattr(self, "c_attention"):
            out = self.c_attention(out)

        out = self.pointwise_conv2(out)

        if hasattr(self, "s_attention"):
            out = self.s_attention(out)

        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        
        return F.relu(out + x)


class DSLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, pooling=True):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), groups=in_channels),
            NormLayer3D(in_channels, norm_type),
        )
    
        if pooling:
            self.pool_layer = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.depthwise_conv(x)

        if hasattr(self, "pool_layer"):
            out = self.pool_layer(out)

        out = self.pointwise_conv(out)
        return out


class DSBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, norm_type, pooling=True):
        super().__init__()
        self.layer = nn.Sequential(
            DSLayer3D(in_channels, hidden_channels, norm_type, pooling=pooling),
            DSLayer3D(hidden_channels, out_channels, norm_type, pooling=False),
        )

        if pooling:
            self.pool_layer = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        if in_channels != out_channels:
            self.adjust_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                NormLayer3D(out_channels, norm_type),
                nn.ReLU(),
            )

    def forward(self, x):
        out = self.layer(x)

        if hasattr(self, "pool_layer"):
            x = self.pool_layer(x)
        
        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        return out + x


class DSBlockCbam(DSBlock):
    def __init__(self, in_channels, hidden_channels, out_channels, norm_type, sq_r, kernel_size, pooling=True):
        super().__init__(in_channels, hidden_channels, out_channels, norm_type, pooling)
        self.c_attn = ChannnelAttention(out_channels, sq_r)
        self.s_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.layer(x)
        out = self.c_attn(out)
        out = self.s_attn(out)

        if hasattr(self, "pool_layer"):
            x = self.pool_layer(x)
        
        if hasattr(self, "adjust_layer"):
            x = self.adjust_layer(x)
        return out + x


class FrontEnd(nn.Module):
    """
    3層の3次元畳み込み
    空間方向に1/4まで圧縮する
    """
    def __init__(self, in_channels, out_channels, dropout, norm_type):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), bias=False)
        self.bn1 = NormLayer3D(32, norm_type)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = NormLayer3D(32, norm_type)
        self.conv3 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = NormLayer3D(out_channels, norm_type)

        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        out : (B, C, H, W, T)
        (H, W)は1/4になります(stride=2が2層あるので) (48, 48) -> (12, 12)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class ResidualBlock3D(nn.Module):
    """
    3次元畳み込み2層を含んだ残差結合ブロック
    """
    def __init__(self, in_channels, out_channels, stride, dropout, norm_type):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=(stride, stride, 1), padding=1, bias=False
        )
        self.bn1 = NormLayer3D(out_channels, norm_type)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = NormLayer3D(out_channels, norm_type)

        if stride > 1 or in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            self.res_bn = NormLayer3D(out_channels, norm_type)

        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        """
        stride=2の場合,(H, W)は1/2になります
        """
        y1 = self.dropout(x)

        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = F.relu(y1)
            
        y1 = self.conv2(y1)
        y1 = self.bn2(y1)

        y2 = x
        if hasattr(self, "res_conv"):
            # 空間方向のstrideが2の場合、空間方向に1/2に圧縮されるのでその分を考慮
            if self.stride > 1:
                y2 = F.avg_pool3d(y2, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
            y2 = self.res_bn(self.res_conv(y2))

        return F.relu(y1 + y2)


class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type) -> None:
        super().__init__()
        # FrondEndを通した時点で(H, W)が(48, 48) -> (12, 12)になるので,層数を制限しています
        assert layers <= 3

        self.frontend = FrontEnd(in_channels, inner_channels, dropout, norm_type)

        res_blocks = []
        res_blocks.append(ResidualBlock3D(
            inner_channels, inner_channels, stride=1, dropout=dropout, norm_type=norm_type
        ))

        # stride=2にすることで空間方向に圧縮する3次元畳み込み
        for _ in range(layers - 1):
            res_blocks.append(ResidualBlock3D(
                inner_channels, inner_channels, stride=2, dropout=dropout, norm_type=norm_type
            ))
        self.res_layers = nn.ModuleList(res_blocks)

        self.out_layer = nn.Conv3d(inner_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x : (B, C, H, W, T)
        out : (B, C, T)
        """
        # 3D convolution & MaxPooling
        out = self.frontend(x)
        
        # residual layers
        for layer in self.res_layers:
            out = layer(out)

        out = self.out_layer(out)

        # W, HについてAverage pooling
        out = torch.mean(out, dim=(2, 3))
        return out


class DSResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            DSBlock(in_channels, inner_channels, inner_channels, norm_type),
            nn.Dropout(dropout),

            DSBlock(inner_channels, inner_channels * 2, inner_channels * 2, norm_type),
            nn.Dropout(dropout),

            DSBlock(inner_channels * 2, inner_channels * 4, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            
            DSBlock(inner_channels * 4, inner_channels * 8, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("DSResNet")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class DSResNet3DCbam(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            DSBlockCbam(in_channels, inner_channels, inner_channels, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels, inner_channels * 2, inner_channels * 2, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels * 2, inner_channels * 4, inner_channels * 4, norm_type, sq_r=16, kernel_size=5),
            nn.Dropout(dropout),
            
            DSBlock(inner_channels * 4, inner_channels * 8, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("DSResNet Cbam")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class DSResNet3DCbamSmall(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            DSBlockCbam(in_channels, inner_channels, inner_channels, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels, inner_channels * 2, inner_channels * 2, norm_type, sq_r=16, kernel_size=7),
            nn.Dropout(dropout),

            DSBlockCbam(inner_channels * 2, inner_channels * 4, inner_channels * 4, norm_type, sq_r=16, kernel_size=5),
            nn.Dropout(dropout),
            
            DSBlock(inner_channels * 4, inner_channels * 8, out_channels, norm_type),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        print("DSResNet Cbam SMall")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        return out


class InvResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            InvResLayer3D(in_channels, inner_channels, norm_type, kernel_size=7),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels, inner_channels + 16, norm_type, kernel_size=7),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels + 16, inner_channels + 32, norm_type, kernel_size=5),
            nn.Dropout(dropout),

            InvResLayer3D(inner_channels + 32, inner_channels + 48, norm_type, s_attn=False),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Conv1d(inner_channels + 48, out_channels, kernel_size=1)

    def forward(self, x):
        print("InvResNet")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


def check_params(net):
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")


if __name__ == "__main__":
    net = ResNet3D(5, 128, 128, 3, 0.1, "bn")
    check_params(net)

    ds_ch = 32
    net = DSResNet3D(5, 128, ds_ch, 3, 0.1, "bn")
    check_params(net)

    net = DSResNet3DCbam(5, 128, ds_ch, 3, 0.1, "bn")
    check_params(net)

    net = DSResNet3DCbamSmall(5, 128, ds_ch, 3, 0.1, "bn")
    check_params(net)

    se_ch = 32
    net = InvResNet3D(5, 128, se_ch, 3, 0.1, "bn")
    check_params(net)