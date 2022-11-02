import pytorch_lightning as pl
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

# TODO: should be relative
default_data_dir = "/n/home11/ccuestalazaro/sunbird/data/"


class DSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        quintile: int = 0,
        multipole: int = 0,
        data_dir: str = default_data_dir,
        batch_size: int = 32,
        standarized: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.quintile = quintile
        self.multipole = multipole
        self.standarized = standarized

    def load_data(self, data_dir, stage):
        # if self.standarized:
        #    return np.load(data_dir / f'{stage}_ds{self.quintile}_m{self.multipole}_standarized.npy')
        return np.load(data_dir / f"{stage}_ds{self.quintile}_m{self.multipole}.npy")

    def load_targets(self, data_dir, stage):
        if self.standarized:
            return np.load(data_dir / f"{stage}_params_standarized.npy")
        return np.load(data_dir / f"{stage}_params.npy")

    def generate_dataset(self, x, y):
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def setup(self, stage: Optional[str] = None):
        self.ds_train = self.generate_dataset(
            self.load_targets(self.data_dir, stage="train"),
            self.load_data(self.data_dir, stage="train"),
        )
        self.ds_test = self.generate_dataset(
            self.load_targets(self.data_dir, stage="test"),
            self.load_data(self.data_dir, stage="test"),
        )
        self.ds_val = self.generate_dataset(
            self.load_targets(self.data_dir, stage="val"),
            self.load_data(self.data_dir, stage="val"),
        )
        self.n_output = self.ds_train.tensors[1].shape[-1]
        self.n_input = self.ds_train.tensors[0].shape[-1]

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
        )
