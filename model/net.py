import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch/model").expanduser()))

import torch
from torch import nn
from torch.nn import functional as F

from spatial_attention import AxialAttentionBlock


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


class Efficient3Dconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 1), groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class SqueezeExcitationLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, up_scale=6, sq_r=12, pooling=True):
        super().__init__()
        self.pooling = pooling
        hidden_channels = int(in_channels * up_scale)
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(hidden_channels, norm_type),
            nn.Hardswish(),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), groups=hidden_channels),
            NormLayer3D(hidden_channels, norm_type),
        )
        self.pooling = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.sq_ex_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // sq_r),
            nn.ReLU(),
            nn.Linear(hidden_channels // sq_r, hidden_channels),
            nn.Sigmoid(),
        )

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv3d(hidden_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )
        self.res_layer = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )

    def forward(self, x):
        res = x

        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)
        
        if self.pooling:
            out = self.pooling(out)

        sq_ex_out = torch.mean(out, dim=(2, 3))     # (B, C, T)
        sq_ex_out = self.sq_ex_layer(sq_ex_out.permute(0, 2, 1)).permute(0, 2, 1)
        sq_ex_out = sq_ex_out.unsqueeze(2).unsqueeze(2)
        out *= sq_ex_out

        out = self.pointwise_conv2(out)
        res = self.res_layer(res)
        return F.hardswish(out + res)


class SqueezeExcitationLayer3DAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, span, n_head=6, up_scale=6, sq_r=12, pooling=True):
        super().__init__()
        self.pooling = pooling
        hidden_channels = int(in_channels * up_scale)
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(hidden_channels, norm_type),
            nn.Hardswish(),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), groups=hidden_channels),
            NormLayer3D(hidden_channels, norm_type),
        )
        self.attention = AxialAttentionBlock(hidden_channels, n_head, span)
        self.pooling = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.sq_ex_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // sq_r),
            nn.ReLU(),
            nn.Linear(hidden_channels // sq_r, hidden_channels),
            nn.Sigmoid(),
        )

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv3d(hidden_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )
        self.res_layer = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )

    def forward(self, x):
        res = x

        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)
        out = self.attention(out)
        
        if self.pooling:
            out = self.pooling(out)

        sq_ex_out = torch.mean(out, dim=(2, 3))     # (B, C, T)
        sq_ex_out = self.sq_ex_layer(sq_ex_out.permute(0, 2, 1)).permute(0, 2, 1)
        sq_ex_out = sq_ex_out.unsqueeze(2).unsqueeze(2)
        out *= sq_ex_out

        out = self.pointwise_conv2(out)
        res = self.res_layer(res)
        return F.hardswish(out + res)


class Pointwise3DConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, up_scale=6):
        super().__init__()
        hidden_channels = int(in_channels * up_scale)
        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(hidden_channels, norm_type),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            NormLayer3D(out_channels, norm_type),
        )
        self.res_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            NormLayer3D(out_channels, norm_type),
        )

    def forward(self, x):
        out = self.pointwise_conv(x)
        res = self.res_layer(x)
        return F.relu(out + res)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, scale=6):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels * scale, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels * scale, in_channels * scale, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), groups=in_channels)
        self.conv3 = nn.Conv3d(in_channels * scale, in_channels * scale, kernel_size=1)
        self.conv4 = nn.Conv3d(in_channels * scale, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class SqueezeExcitationLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type, scale=6, sq_factor=12):
        super().__init__()
        self.stride = stride
        hidden_channels = in_channels * scale

        self.pointwise_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            NormLayer3D(hidden_channels, norm_type),
            nn.ReLU(),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), groups=in_channels),
            NormLayer3D(hidden_channels, norm_type),
        )

        self.sq_ex_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // sq_factor),
            nn.ReLU(),
            nn.Linear(hidden_channels // sq_factor, hidden_channels),
            nn.Sigmoid(),
        )

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
            NormLayer3D(out_channels, norm_type),
        )
        self.res_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            NormLayer3D(out_channels, norm_type),
        )

    def forward(self, x):
        res = x
        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)

        sq = torch.mean(out, dim=(2, 3))    # (B, C, T)
        sq = self.sq_ex_layer(sq.permute(0, 2, 1)).permute(0, 2, 1)
        sq = sq.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        out *= sq

        out = self.pointwise_conv2(out)
        
        if self.stride > 1:
            res = F.max_pool3d(res, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        res = self.res_layer(res)
        
        return F.relu(out + res)


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


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout, norm_type):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), bias=False)
        self.bn1 = NormLayer3D(out_channels, norm_type)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False)
        self.bn2 = NormLayer3D(out_channels, norm_type)

        if stride > 1 or in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            self.res_bn = NormLayer3D(out_channels, norm_type)

        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        y1 = x
        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = F.relu(y1)
            
        y1 = self.conv2(y1)
        y1 = self.bn2(y1)

        y2 = x
        if hasattr(self, "res_conv"):
            if self.stride > 1:
                y2 = F.avg_pool3d(y2, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
            y2 = self.res_bn(self.res_conv(y2))

        return self.dropout(F.relu(y1 + y2))


class EfficientResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, inner_channels, kernel_size=(5, 5, 5), stride=(2, 2, 1), padding=(2, 2, 2)),
            NormLayer3D(inner_channels, norm_type),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        in_cs = [inner_channels, inner_channels, inner_channels * 2]
        out_cs = [inner_channels, inner_channels * 2, inner_channels*4]
        stride = [1, 2, 2]
        self.conv2d_layers = nn.ModuleList([
            nn.Sequential(
                ResBlock2D(in_c, out_c, s, dropout, norm_type),
            ) for in_c, out_c, s in zip(in_cs, out_cs, stride)
        ])
        self.out_layer = nn.Conv1d(out_cs[-1], out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        out : (B, C, T)
        """
        print("eff")
        out = self.conv3d(x)
        out = self.max_pool(out)
        for layer in self.conv2d_layers:
            out = layer(out)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class MoreEfficientResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            # nn.Conv3d(in_channels, 32, kernel_size=(5, 5, 5), stride=(2, 2, 1), padding=(2, 2, 2)),   # 1つめ
            # NormLayer3D(32, norm_type),
            # nn.ReLU(),
            # SqueezeExcitationLayer3D(32, inner_channels * 2, norm_type),
            SqueezeExcitationLayer3D(in_channels, inner_channels, norm_type),   # 2つめ
            SqueezeExcitationLayer3D(inner_channels, inner_channels * 2, norm_type),
        )

        in_cs = [inner_channels * 2, inner_channels * 2, inner_channels * 4]
        out_cs = [inner_channels * 2, inner_channels * 4, inner_channels * 8]
        stride = [1, 2, 2]
        self.conv2d_layers = nn.ModuleList([
            nn.Sequential(
                SqueezeExcitationLayer2D(in_c, out_c, s, norm_type)
            ) for in_c, out_c, s in zip(in_cs, out_cs, stride)
        ])
        self.out_layer = nn.Conv1d(out_cs[-1], out_channels, kernel_size=1)

    def forward(self, x):
        print("more eff")
        out = self.conv3d(x)
        for layer in self.conv2d_layers:
            out = layer(out)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class MoreEfficientResNetAll3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            SqueezeExcitationLayer3D(in_channels, inner_channels, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels, inner_channels * 2, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 2, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 4, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("more eff all_3d")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class MoreEfficientResNetAll3DAttention(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            SqueezeExcitationLayer3DAttention(in_channels, inner_channels, norm_type, span=48, n_head=5),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3DAttention(inner_channels, inner_channels * 2, norm_type, span=24),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 2, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 4, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("more eff all_3d attention")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class MoreEfficientResNetAll3D_Bigger(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            SqueezeExcitationLayer3D(in_channels, inner_channels, norm_type),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels, inner_channels, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels, inner_channels * 2, norm_type),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels * 2, inner_channels * 2, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 2, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels * 4, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 4, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels * 8, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("more eff all_3d bigger")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out

    
class MoreEfficientResNetAll3D_BiggerAttention(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            SqueezeExcitationLayer3DAttention(in_channels, inner_channels, norm_type, span=48, n_head=5),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels, inner_channels, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3DAttention(inner_channels, inner_channels * 2, norm_type, span=24),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels * 2, inner_channels * 2, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3DAttention(inner_channels * 2, inner_channels * 4, norm_type, span=12),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels * 4, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 4, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
            Pointwise3DConvLayer(inner_channels * 8, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("more eff all_3d bigger")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


class MoreEfficientResNetAll3D_Bigger2(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
        super().__init__()
        self.conv3d = nn.Sequential(
            SqueezeExcitationLayer3D(in_channels, inner_channels, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels, inner_channels, norm_type, pooling=False),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels, inner_channels * 2, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 2, inner_channels * 2, norm_type, pooling=False),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 2, inner_channels * 4, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 4, inner_channels * 4, norm_type, pooling=False),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 4, inner_channels * 8, norm_type),
            nn.Dropout(dropout),
            SqueezeExcitationLayer3D(inner_channels * 8, inner_channels * 8, norm_type, pooling=False),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Conv1d(inner_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        print("more eff all_3d bigger2")
        out = self.conv3d(x)
        out = torch.mean(out, dim=(2, 3))
        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    print("ResNet3D default")
    net = ResNet3D(5, 128, 64, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("EfficientResNet3D")
    net = EfficientResNet3D(5, 128, 32, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("MoreEfficientResNet3D")
    net = MoreEfficientResNet3D(5, 128, 16, 3, 0.1, norm_type="bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("MoreEfficientResNetAll3D")
    net = MoreEfficientResNetAll3D(5, 128, 16, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("MoreEff all_3d attention")
    net = MoreEfficientResNetAll3DAttention(5, 128, 16, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("Moreeff all_3d bigger")
    net = MoreEfficientResNetAll3D_Bigger(5, 128, 16, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("Moreeff all_3d bigger2")
    net = MoreEfficientResNetAll3D_Bigger2(5, 128, 16, 3, 0.1, "bn")
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("normal 3d convolution")
    net = nn.Conv3d(64, 64, kernel_size=3, padding=1)
    x = torch.rand(1, 64, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("sq_ex 3d")
    net = SqueezeExcitationLayer3D(64, 64, "bn")
    x = torch.rand(1, 64, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("pointwise 3d")
    net = Pointwise3DConvLayer(64, 64, "bn")
    x = torch.rand(1, 64, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("normal 2d convolution")
    net = nn.Conv3d(64, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0))
    x = torch.rand(1, 64, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print(f"depthwise separable conv")
    net = DepthwiseSeparableConv(64, 128, stride=1)
    x = torch.rand(1, 64, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print(f"inverted residual block")
    net = InvertedResidualBlock(8, 16, stride=1, scale=6)
    x = torch.rand(1, 8, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print(f"squeeze and excitation")
    net = SqueezeExcitationLayer2D(8, 16, stride=1, norm_type="bn", scale=6, sq_factor=12)
    x = torch.rand(1, 8, 48, 48, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    print("AxialAttention")
    d_model = 16 * 6
    n_head = 6
    span = 48

    net = AxialAttentionBlock(d_model, n_head, span)
    x = torch.rand(1, d_model, span, span, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")