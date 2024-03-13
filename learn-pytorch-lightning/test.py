import datetime
import os

import dotenv
import hydra
import lightning as L
import omegaconf
import torch
import torch.utils.data as data
from datamodules import MNISTDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models import Decoder, Encoder, LitAutoEncoder
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

wandb.login(key=os.environ["WANDB_API_KEY"])
dotenv.load_dotenv(dotenv_path="../.env")
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def main_without_dm(cfg: omegaconf.DictConfig):
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

    model = LitAutoEncoder(Encoder(cfg), Decoder(cfg), cfg)

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
        max_epochs=cfg["app"]["max_epoch"],
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg["app"]["early_stopping_patience"],
            ),
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
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    trainer.test(model=model, dataloaders=test_loader)


def main_with_dm(cfg: omegaconf.DictConfig):
    torch.set_float32_matmul_precision("medium")

    datamodule = MNISTDataModule(cfg)
    model = LitAutoEncoder(Encoder(cfg), Decoder(cfg), cfg)
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
    wandb_logger.watch(model, log="all")
    trainer = L.Trainer(
        default_root_dir="./",
        max_epochs=cfg["app"]["max_epoch"],
        callbacks=[
            EarlyStopping(
                monitor=cfg["app"]["monitoring_metric"],
                mode=cfg["app"]["monitoring_mode"],
                patience=cfg["app"]["early_stopping_patience"],
            ),
            ModelCheckpoint(
                monitor=cfg["app"]["monitoring_metric"],
                mode=cfg["app"]["monitoring_mode"],
                every_n_epochs=cfg["app"]["save_checkpoint_every_n_epochs"],
                save_top_k=cfg["app"]["save_checkpoint_top_k"],
                dirpath=f"./pytorch-lightning-test/{now}",
                filename="{epoch}-{step}-{val_loss:.3f}",
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        limit_train_batches=float(cfg["app"]["limit_train_batches"]),
        limit_val_batches=float(cfg["app"]["limit_val_batches"]),
        limit_test_batches=float(cfg["app"]["limit_test_batches"]),
        profiler=None,
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        log_every_n_steps=cfg["app"]["log_every_n_steps"],
        precision=cfg["app"]["precision"],
        accumulate_grad_batches=cfg["app"]["accumulate_grad_batches"],
        gradient_clip_val=cfg["app"]["gradient_clip_val"],
        gradient_clip_algorithm=cfg["app"]["gradient_clip_algorithm"],
    )

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model=model, datamodule=datamodule, mode="power")
    # tuner.lr_find(model=model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")
    preds = trainer.predict(datamodule=datamodule, ckpt_path="best")


def load_checkpoint(checkpoint_path: str, cfg: omegaconf.OmegaConf):
    model = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        encoder=Encoder(cfg),
        decoder=Decoder(cfg),
    )
    model.eval()


@hydra.main(version_base=None, config_path="./conf-2", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    main_with_dm(cfg)


if __name__ == "__main__":
    main()
