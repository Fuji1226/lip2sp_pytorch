import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
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
        relative_idx = key_idx - query_idx + span - 1
        self.relative_idx = relative_idx
        self.register_buffer("flatten_idx", relative_idx.view(-1))

        self.pooling = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        """
        x : (B, C, H, W, T)
        """
        res = x
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

        qr = torch.einsum("bhci,cij->bhij", query, q_embedding)
        kr = torch.einsum("bhci,cij->bhij", key, k_embedding)
        qk = torch.einsum("bhci,bhcj->bhij", query, key)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = stacked_similarity.view(B * T * W, 3, self.n_head, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)

        sv = torch.einsum('bhij,bhcj->bhci', similarity, value)
        sve = torch.einsum('bhij,cij->bhci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(B * T * W, self.d_model * 2, H)
        output = self.bn_output(stacked_output).view(B, T, W, self.d_model, 2, H).sum(dim=-2)
        
        if self.width:
            output = output.permute(0, 3, 2, -1, 1)
        else:
            output = output.permute(0, 3, -1, 2, 1)

        return F.relu(output + res)

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

    def forward(self, x):
        out = self.attention(x)
        return out


if __name__ == "__main__":
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