import lightning as L
import torch
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import nn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, cfg["app"]["hidden_channels"]),
            nn.ReLU(),
            nn.Linear(cfg["app"]["hidden_channels"], 3),
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, cfg["app"]["hidden_channels"]),
            nn.ReLU(),
            nn.Linear(cfg["app"]["hidden_channels"], 28 * 28),
        )

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, cfg):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg
        self.learning_rate = cfg["app"]["learning_rate"]  # for Tuner
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.example_input_array = self.example_input_array.view(
            self.example_input_array.size(0), -1
        )
        self.automatic_optimization = True

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98)
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.cfg['app']['max_epoch'],
            cycle_mult=1.0,
            max_lr=1.0e-3,
            min_lr=1.e-8,
            warmup_steps=self.cfg['app']['max_epoch'] // 10,
            gamma=1.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
