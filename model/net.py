import torch
from torch import nn
from torch.nn import functional as F


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
            y2 = F.avg_pool3d(y2, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
            y2 = self.res_bn(self.res_conv(y2))

        return F.relu(y1 + y2)


class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type) -> None:
        super().__init__()
        # FrondEndを通した時点で(H, W)が(48, 48) -> (12, 12)になるので,層数を制限しています
        assert layers <= 4

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


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout) -> None:
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#             nn.Dropout3d(dropout),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#             nn.Dropout3d(dropout),
#         )

#     def forward(self, x):
#         res = x
#         out = self.layers(x) + res
#         return out


# class ResNet3D2(nn.Module):
#     def __init__(self, in_channels, out_channels, inner_channels, layers, dropout, norm_type):
#         super().__init__()
#         assert layers <= 3
#         in_cs = [(inner_channels // 2) // (2**i) for i in range(layers)]
#         out_cs = [inner_channels // (2**i) for i in range(layers)]
#         in_cs.append(in_channels)
#         out_cs.append(in_cs[-2])
#         in_cs = list(reversed(in_cs))
#         out_cs = list(reversed(out_cs))

#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1), bias=False),
#                 nn.BatchNorm3d(out_c),
#                 nn.ReLU(),
#                 ResBlock(out_c, out_c, dropout),
#             ) for in_c, out_c in zip(in_cs, out_cs)
#         ])
#         self.out_layer = nn.Conv1d(inner_channels, out_channels, kernel_size=1)
    
#     def forward(self, x):
#         """
#         x : (B, C, H, W, T)
#         out : (B, C, T)
#         """
#         out = x
#         for layer in self.layers:
#             out = layer(out)

#         # W, HについてAverage pooling
#         out = torch.mean(out, dim=(2, 3))
#         out = self.out_layer(out)
#         return out


if __name__ == "__main__":
    net = ResNet3D(5, 256, 128, 3, 0.1)
    x = torch.rand(1, 5, 48, 48, 150)
    out = net(x)

    x = torch.rand(1, 5, 48, 48, 150)
    norm_layer = NormLayer3D(5, norm_type="bn")
    out = norm_layer(x)
    print(out.shape)
