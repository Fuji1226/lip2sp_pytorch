import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureQuantizerEMA(nn.Module):
    """
    Feature Quantization modules using exponential moving average.
    This modules follow the equation (8) in the original paper.
    https://github.com/YangNaruto/FQ-GAN/blob/master/FQ-BigGAN/vq_layer.py
    Args:
        emb_dim (int): Size of feature vector on dictionary.
        num_emb (int): Number of feature vector on dictionary.
        commitment (float): Strength of commitment loss. Defaults to 0.25.
        decay (float, optional): Moment coefficient. Defaults to 0.9.
        eps (float, optional): sufficient small value to avoid dividing by zero. Defaults to 1e-5.
    """
    def __init__(self,
                 emb_dim,
                 num_emb,
                 commitment=0.25,
                 decay=0.9,
                 eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.commitment = commitment
        self.decay = decay
        self.eps = eps

        embed = torch.randn(self.emb_dim, self.num_emb)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_emb))
        self.register_buffer("ema_embed", embed.clone())

    def forward(self, x):
        r"""
        Feature quantization feedforward function.
        Args:
            x (Tensor): Input feature map.
        Returns:
            Tensor: Output quantized feature map.
            Tensor: Loss
            Tensor: Embedding index for reference of shape [B, H, W]
        """
        # x: [B, C=D, H, W] --> [B, H, W, C=D]
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.size()

        # flatten: [B, H, W, D] --> [N(=B x H x W), D]
        flatten = x.view(-1, self.emb_dim)

        # distance: d(flatten[N, D], embed[D, K]) --> [N, K]
        distance = (
            flatten.pow(2).sum(dim=1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(dim=0, keepdim=True)
        )

        # embed_idx: [N, K] --> [N, ]
        embed_idx = torch.argmin(distance, dim=1)

        # set onehot label: [N, ] --> [N, K]
        embed_onehot = F.one_hot(embed_idx, num_classes=self.num_emb).type(flatten.dtype)

        # embed_idx: [N, ] --> [B, H, W, ]
        embed_idx = embed_idx.view(input_shape[:-1])

        # quantize: [B, H, W, ] embed [K, D] --> [B, H, W, D]
        quantize = F.embedding(embed_idx, self.embed.transpose(0, 1))

        # train embedding vector only when model.train(), not model.eval()
        if self.training:
            # ref_count: [N, K] --> [K, ]
            ref_count = torch.sum(embed_onehot, dim=0)

            # ema for reference count: [K, ] by N = decay * N + (1 - decay) * n
            self.cluster_size.data.mul_(self.decay).add_(
                ref_count, alpha=1 - self.decay
            )

            # total reference count
            n = self.cluster_size.sum()

            # additive (or, laplace) smoothing
            smoothing_cluster_size = n * (
                (self.cluster_size + self.eps) / (n + self.cluster_size * self.eps)
            )

            # dw: [D, N] @ [N, K]
            dw = flatten.transpose(0, 1) @ embed_onehot

            # ema for embedding: [D, K]
            self.ema_embed.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

            # normalize: [D, K] / [1, K] --> [K, ]
            embed_norm = self.ema_embed / smoothing_cluster_size.unsqueeze(0)

            # embed = self.ema_embed / self.cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_norm)

        # loss
        e_latent_loss = F.mse_loss(quantize.detach(), x)
        loss = self.commitment * e_latent_loss

        # straight through estimator
        quantize = x + (quantize - x).detach()

        # quantize: [B, H, W, D] --> [B, D, H, W]
        quantize = quantize.permute(0, 3, 1, 2).contiguous()

        return quantize, loss, embed_idx

    def extra_repr(self):
        return "emb_dim={}, num_emb={}, commitment={}, decay={}, eps={}".format(
            self.emb_dim, self.num_emb, self.commitment, self.decay, self.eps
        )


class VQ(nn.Module):
    def __init__(self, emb_dim, num_emb, commitment=0.25, decay=0.9, eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.commitment = commitment
        self.decay = decay
        self.eps = eps

        embed = torch.randn(self.emb_dim, self.num_emb)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_emb))
        self.register_buffer("ema_embed", embed.clone())

    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = x.permute(0, 2, 1).contiguous()  # (B, T, C)
        input_shape = x.size()
        flatten = x.view(-1, self.emb_dim)

        # distance: d(flatten[N, D], embed[D, K]) --> [N, K]
        distance = (
            flatten.pow(2).sum(dim=1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(dim=0, keepdim=True)
        )

        # embed_idx: [N, K] --> [N, ]
        embed_idx = torch.argmin(distance, dim=1)

        # set onehot label: [N, ] --> [N, K]
        embed_onehot = F.one_hot(embed_idx, num_classes=self.num_emb).type(flatten.dtype)

        # embed_idx: [N, ] --> [B, T, ]
        embed_idx = embed_idx.view(input_shape[:-1])

        # quantize: [B, T, ] embed [K, D] --> [B, T, D]
        quantize = F.embedding(embed_idx, self.embed.transpose(0, 1))

        # train embedding vector only when model.train(), not model.eval()
        if self.training:
            # ref_count: [N, K] --> [K, ]
            ref_count = torch.sum(embed_onehot, dim=0)

            # ema for reference count: [K, ] by N = decay * N + (1 - decay) * n
            self.cluster_size.data.mul_(self.decay).add_(
                ref_count, alpha=1 - self.decay
            )

            # total reference count
            n = self.cluster_size.sum()

            # additive (or, laplace) smoothing
            smoothing_cluster_size = n * (
                (self.cluster_size + self.eps) / (n + self.cluster_size * self.eps)
            )

            # dw: [D, N] @ [N, K]
            dw = flatten.transpose(0, 1) @ embed_onehot

            # ema for embedding: [D, K]
            self.ema_embed.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

            # normalize: [D, K] / [1, K] --> [K, ]
            embed_norm = self.ema_embed / smoothing_cluster_size.unsqueeze(0)

            # embed = self.ema_embed / self.cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_norm)

        # loss
        e_latent_loss = F.mse_loss(quantize.detach(), x)
        loss = self.commitment * e_latent_loss

        # straight through estimator
        quantize = x + (quantize - x).detach()

        quantize = quantize.permute(0, 2, 1).contiguous()
        return quantize, loss, embed_idx


if __name__ == "__main__":
    net = VQ(emb_dim=64, num_emb=50)
    net.train()
    print(net.embed)
    x = torch.rand(1, 64, 150)
    quantize, loss, embed_idx = net(x)
    print(net.embed)
    breakpoint()