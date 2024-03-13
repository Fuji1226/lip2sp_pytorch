import lightning as L
import omegaconf
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["app"]["batch_size"]  # for Tuner
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.cfg["app"]["data_dir"], train=True, download=True)
        MNIST(self.cfg["app"]["data_dir"], train=False, download=True)

    def setup(self, stage: str):
        """
        build dataset instances
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.cfg["app"]["data_dir"], train=True, transform=self.transform
            )
            train_set_size = int(len(mnist_full) * self.cfg["app"]["train_ratio"])
            valid_set_size = len(mnist_full) - train_set_size
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(self.cfg["app"]["seed"]),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.cfg["app"]["data_dir"], train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.cfg["app"]["data_dir"], train=False, transform=self.transform
            )

    def train_dataloader(self):
        """
        Use the train_dataloader() method to generate the training dataloader(s).
        Usually you just wrap the dataset you defined in setup.
        This is the dataloader that the Trainer fit() method uses.
        """
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.cfg["app"]["num_workers"],
        )

    def val_dataloader(self):
        """
        fit() and validate()
        """
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.cfg["app"]["num_workers"],
        )

    def test_dataloader(self):
        """
        test()
        """
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.cfg["app"]["num_workers"],
        )

    def predict_dataloader(self):
        """
        predict()
        """
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.cfg["app"]["num_workers"],
        )
