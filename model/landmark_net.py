import torch
import torch.nn as nn


class LandmarkConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, dropout, aggregate_xy):
        super().__init__()
        if aggregate_xy:
            self.agg_fc = nn.Sequential(
                nn.Linear(2, 1),
                nn.ReLU(),
            )

        layers = []
        for i in range(n_layers):
            if i == 0:
                in_c = in_channels
            else:
                in_c = out_channels
            padding = (kernel_size - 1) // 2

            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.LayerNorm((out_channels, 150)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, C, T) or (B, C, 2, T)
        """
        if hasattr(self, "agg_fc"):
            x = self.agg_fc(x.permute(0, 1, 3, 2)).squeeze(-1)

        for layers in self.layers:
            x = layers(x)
        return x


class LMCoProcessingNet(nn.Module):
    def __init__(self, out_channels, kernel_size, n_layers, dropout, compress_time_axis):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                in_c = 2
                if compress_time_axis:
                    stride = 2
                else:
                    stride = 1
            else:
                in_c = out_channels
                stride = 1
            padding = (kernel_size - 1) // 2

            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x : (B, 2, K, T)
        output : (B, C, K, T)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, int(d_model * 4)),
            nn.ReLU(),
            nn.Linear(int(d_model * 4), d_model),
            nn.Dropout(dropout),
        )
        self.bn = nn.BatchNorm2d(d_model)

    def forward(self, x):
        """
        x : (B, T, K, C)
        output : (B, T, K, C)
        """
        output = self.layers(x) + x
        output = self.bn(output.permute(0, 3, 2, 1))    # (B, C, K, T)
        return output.permute(0, 3, 2, 1)


class ASTT_GCN(nn.Module):
    def __init__(self, d_model, n_head, n_nodes, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_channels = d_model // n_head

        self.query = nn.Linear(self.head_channels, self.head_channels)
        self.key_layer = nn.Linear(self.head_channels, self.head_channels)
        self.node_feature_layer = nn.Linear(self.head_channels, self.head_channels)

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(d_model)

        self.out_layer = LinearBlock(d_model, dropout)

        a_se = torch.ones(1, 1, 1, n_nodes, n_nodes) / (10 ** 6)
        self.a_se = nn.parameter.Parameter(a_se, requires_grad=True)

    def forward(self, x):
        """
        x : (B, T, K, C)
        output : (B, T, K, C)
        """
        B, T, K, C = x.shape
        res = x
        x = x.reshape(B, T, K, self.n_head, -1)
        query = self.query(x).permute(0, 1, 3, 2, 4)    # (B, T, n_head, K, C)
        key = self.key_layer(x).permute(0, 1, 3, 4, 2)  # (B, T, n_head, C, K)
        
        a_sa = torch.matmul(query, key)     # (B, T, n_head, K, K)
        a_sa = torch.softmax(a_sa, dim=-1)
        a_all = a_sa + self.a_se

        output = self.node_feature_layer(x).permute(0, 1, 3, 2, 4)      # (B, T, n_head, K, C)
        output = torch.matmul(a_all, output)

        output = output.permute(0, 1, 3, 2, 4).reshape(B, T, K, C)
        output = self.dropout(output)
        output = output + res
        output = output.permute(0, 3, 2, 1)     # (B, C, K, T)
        output = self.bn(output).permute(0, 3, 2, 1)    # (B, T, K, C)

        output = self.out_layer(output)
        return output


class LandmarkEncoder(nn.Module):
    def __init__(self, inner_channels, lmco_kernel_size, lmco_n_layers, compress_time_axis, astt_gcn_n_layers, astt_gcn_n_head, n_nodes, dropout):
        super().__init__()
        self.lmco = LMCoProcessingNet(
            out_channels=inner_channels,
            kernel_size=lmco_kernel_size,
            n_layers=lmco_n_layers,
            dropout=dropout,
            compress_time_axis=compress_time_axis,
        )
        
        self.semantic_encoding = nn.Embedding(n_nodes, inner_channels)

        astt_gcns = []
        for i in range(astt_gcn_n_layers):
            astt_gcns.append(
                ASTT_GCN(
                    d_model=inner_channels,
                    n_head=astt_gcn_n_head,
                    n_nodes=n_nodes,
                    dropout=dropout,
                )
            )

        self.astt_gcns = nn.ModuleList(astt_gcns)

    def forward(self, x, landmark_index):
        """
        x : (B, 2, K, T)
        landmark_index : (B, K)
        output : (B, T, C)
        """
        output = self.lmco(x)   # (B, C, K, T)
        se_emb = self.semantic_encoding(landmark_index).permute(0, 2, 1).unsqueeze(-1)     # (B, C, K, 1)
        output = output + se_emb
        output = output.permute(0, 3, 2, 1)     # (B, T, K, C)

        for layer in self.astt_gcns:
            output = layer(output)
        
        output = torch.mean(output, dim=2)  # (B, T, C)
        return output   


if __name__ == "__main__":
    B = 16
    T = 150
    K = 38
    C = 64
    x = torch.rand(B, 2, K, T)
    landmark_index = torch.tensor([i for i in range(K)]).unsqueeze(0).expand(B, -1)

    net = LandmarkEncoder(
        inner_channels=C,
        lmco_kernel_size=3,
        lmco_n_layers=3,
        compress_time_axis=True,
        astt_gcn_n_layers=2,
        astt_gcn_n_head=C // 64,
        n_nodes=K,
        dropout=0.1,
    )
    output = net(x, landmark_index)
    print(output.shape)

    x = torch.rand(B, int(K * 2), T)
    net = LandmarkConv(
        in_channels=int(K * 2),
        out_channels=256,
        kernel_size=3,
        n_layers=5,
        dropout=0.1,
        aggregate_xy=False,
    )
    out = net(x)
    print(out.shape)

    x = torch.rand(B, K, 2, T)
    net = LandmarkConv(
        in_channels=K,
        out_channels=256,
        kernel_size=3,
        n_layers=5,
        dropout=0.1,
        aggregate_xy=True,
    )
    out = net(x)
    print(out.shape)