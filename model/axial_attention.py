import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
    """
    2次元データに対して縦横1次元ずつself attentionを行う
    まずどちらかの軸に対してattentionを行い,その出力に対しての別の軸からattentionを行うことで,結果的に2次元平面全体を考慮したattentionが行われるという考え
    2次元データに対してそのままself attentionを行うと計算量が(hw)**2になってしまうが,これだとhw(h + w)に抑えることができ,3乗のオーダーになるのでメモリが節約できる
    また,attentionの際にはqueryとkey,valueの相対的な位置関係を考慮する
    """
    def __init__(self, d_model, n_head, span, stride, width=False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_channels = d_model // n_head
        self.span = span
        self.width = width

        self.qkv_transform = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=False)
        self.bn_qkv = nn.BatchNorm1d(d_model * 2)
        self.bn_similarity = nn.BatchNorm2d(n_head * 3)
        self.bn_output = nn.BatchNorm1d(d_model * 2)

        self.relative = nn.Parameter(torch.randn(self.head_channels * 2, span * 2 - 1), requires_grad=True)
        query_idx = torch.arange(span).unsqueeze(0)
        key_idx = torch.arange(span).unsqueeze(1)
        # relative_idx = key_idx - query_idx + span - 1
        relative_idx = key_idx + query_idx

        # print(relative_idx)
        # relative_idxは対角線が同じ値になっているが、おそらく問題ない
        # 例
        # 0, 1, 2, 3
        # 1, 2, 3, 4
        # 2, 3, 4, 5
        # 3, 4, 5, 6

        # axial attentionでは2次元データに対して一気にattentionを計算するのではなく、縦横1次元ずつ行うことで結果的に2次元全体の情報が伝播される仕組みになっている
        # なので、attention計算における相対的な位置関係は、self attentionを計算する軸ごとにあればよく、またその軸同士の位置関係が与えられていれば十分
        # 例えば1つ目の軸についてself attentionを行うときのqueryとkeyの相対的な位置関係は、インデックスの行方向の関係にあたる
        # 一方、1つ目の軸と別の軸との相対的な位置関係は、インデックスの列方向の関係で考慮されるはず
        # なので、(0, 1)と(1, 0)が同じ値になっていても、これらはattentionにおける相対的な位置関係を考慮する上では問題にならない気がする

        self.register_buffer("flatten_idx", relative_idx.view(-1))
        self.pooling = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        if self.width:
            x = x.permute(0, -1, 2, 1, 3)   # (B, T, H, C, W)
        else:
            x = x.permute(0, -1, 3, 1, 2)   # (B, T, W, C, H)
        B, T, W, C, H = x.shape
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        qkv = self.qkv_transform(x)
        query, key, value = torch.split(qkv.reshape(B * T * W, self.n_head, self.head_channels * 2, H), [self.head_channels // 2, self.head_channels // 2, self.head_channels], dim=2)

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_idx).view(self.head_channels * 2, self.span, self.span)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.head_channels // 2, self.head_channels // 2, self.head_channels], dim=0)

        # queryからみたkeyとの相対的な位置関係の考慮
        qr = torch.einsum("bhci,cij->bhij", query, q_embedding)

        # keyから見たqueryとの相対的な位置関係の考慮
        kr = torch.einsum("bhci,cij->bhij", key, k_embedding)

        # print(f"q = {query.shape}, q_embedding = {q_embedding.shape}, qr = {qr.shape}")

        # 上の計算は下4行と同じ計算をしている
        # まず2つの行列の共通しない次元は、unsqueezeされる
        # これにより2つの行列の形状が揃い、要素積が計算される
        # ->で記す計算結果の形状についてなくなっている次元（上だとc）は、sumでその次元について和をとることに相当する
        # query = query.unsqueeze(-1).expand(-1, -1, -1, -1, q_embedding.shape[-1])     # (B * T * W, n_head, C, H, H)
        # q_embedding = q_embedding.unsqueeze(0).unsqueeze(0).expand_as(query)      # (B * T * W, n_head, C, H, H)
        # ans = query * q_embedding
        # ans = ans.sum(2)      # (B * T * W, n_head, H, H)

        # queryとkeyの関係性
        qk = torch.einsum("bhci,bhcj->bhij", query, key)

        # 上の計算も同じように共通しない次元が拡張され、要素積を取った後に和を取ってまとめている
        # こっちは(i, c)と(c, j)で行列積を取っていると考えた方がわかりやすいかも
        # query = query.unsqueeze(-1)
        # key = key.unsqueeze(-2)
        # ans = query * key
        # ans = ans.sum(2)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = stacked_similarity.view(B * T * W, 3, self.n_head, H, H).sum(dim=1)    # (B * T * W, n_head, H, H)
        similarity = F.softmax(stacked_similarity, dim=-1)

        sv = torch.einsum('bhij,bhcj->bhci', similarity, value)
        sve = torch.einsum('bhij,cij->bhci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(B * T * W, self.d_model * 2, H)
        output = self.bn_output(stacked_output)
        output = output.view(B, T, W, self.d_model, 2, H).sum(dim=-2)
        
        if self.width:
            output = output.permute(0, 3, 2, -1, 1)
        else:
            output = output.permute(0, 3, -1, 2, 1)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.d_model))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / (self.head_channels)))


class AxialAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, span, stride=1):
        super().__init__()
        self.attention = nn.Sequential(
            AxialAttention(d_model, n_head, span, stride, width=False),
            AxialAttention(d_model, n_head, span, stride, width=True),
        )
        self.fc = nn.Conv3d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        out = self.attention(x)
        out = F.relu(self.fc(out))
        return out + x


if __name__ == "__main__":
    d_model = 32
    n_head = 2
    span = 12

    net = AxialAttentionBlock(d_model, n_head, span)
    x = torch.rand(1, d_model, span, span, 150)
    out = net(x)
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"out = {out.shape}, params = {params}")

    x = torch.arange(2).reshape(1, 2)
    y = torch.arange(4).reshape(1, 2, 2)
    ans = torch.einsum("ab,abc->bc", x, y)

    span = 6
    x = torch.arange(2 * span).reshape(2, span)
    y = torch.arange(2 * span * span).reshape(2, span, span)
    ans = torch.einsum("ab,abc->bc", x, y)

    x = torch.arange(2 * span).reshape(2, span).unsqueeze(2) 
    y = torch.arange(2 * span * span).reshape(2, span, span)
    ans = x * y
    ans = ans.sum(0)