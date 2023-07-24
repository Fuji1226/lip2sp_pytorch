import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.fc = nn.Linear(in_channels, in_channels)
        self.fc_mu = nn.Linear(in_channels, latent_dim)
        self.fc_logvar = nn.Linear(in_channels, latent_dim)

    def forward(self, x):
        """
        x : (B, T, C)
        mu, logvar, z : (B, T, C)
        """
        h = torch.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        eps = torch.randn_like(torch.exp(logvar))
        z = mu + torch.exp(logvar / 2) * eps
        return mu, logvar, z