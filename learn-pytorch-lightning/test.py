import os

import dotenv
import hydra
import lightning as L
import omegaconf
import torch
import torch.nn.functional as F
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

import wandb

dotenv.load_dotenv(dotenv_path="../.env")
wandb.login(key=os.environ["WANDB_API_KEY"])


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


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
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.example_input_array = self.example_input_array.view(
            self.example_input_array.size(0), -1
        )

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg["app"]["learning_rate"]
        )
        return optimizer


@hydra.main(version_base=None, config_path="./conf-2", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root="MNIST", download=True, train=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="MNIST", download=True, train=False, transform=transform
    )

    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(
        train_set, [train_set_size, valid_set_size], generator=seed
    )

    train_loader = DataLoader(
        train_set,
        num_workers=cfg["app"]["num_workers"],
        batch_size=cfg["app"]["batch_size"],
    )
    valid_loader = DataLoader(
        valid_set,
        num_workers=cfg["app"]["num_workers"],
        batch_size=cfg["app"]["batch_size"],
    )
    test_loader = DataLoader(test_set, num_workers=cfg["app"]["num_workers"])

    autoencoder = LitAutoEncoder(Encoder(cfg), Decoder(cfg), cfg)

    wandb_logger = WandbLogger(
        project="pytorch-lightning-test",
        # name="test",
        log_model=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
        # group="group-test-sweep-1",
    )

    trainer = L.Trainer(
        default_root_dir="./",
        max_epochs=20,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            ModelCheckpoint(monitor="val_loss", mode="min", save_last=True),
        ],
        # fast_dev_run=10,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        profiler=None,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
    )
    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # trainer.test(model=autoencoder, dataloaders=test_loader)

    # autoencoder = LitAutoEncoder.load_from_checkpoint(
    #     "/home/minami/lip2sp_pytorch/learn-pytorch-lightning/lightning_logs/version_0/checkpoints/epoch=0-step=48000.ckpt",
    #     encoder=Encoder(),
    #     decoder=Decoder(),
    # )
    # autoencoder.eval()


if __name__ == "__main__":
    main()
