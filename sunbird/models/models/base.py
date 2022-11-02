import torch
import numpy as np
from pathlib import Path
import yaml
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_folder(cls, path_to_model: str):
        path_to_model = Path(path_to_model)
        with open(path_to_model / "hparams.yaml") as f:
            hparams = yaml.safe_load(f)
        model = cls(**hparams)
        # find file with lowest validation loss
        files = list((path_to_model / "checkpoints").glob("*.ckpt"))
        file_idx = np.argmin(
            [float(str(file).split(".ckpt")[0].split("=")[-1]) for file in files]
        )
        weights_dict = torch.load(
            files[file_idx],
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(weights_dict["state_dict"])
        return model

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch=batch, batch_idx=batch_idx)
        self.log(
            "train_loss",
            loss,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch=batch, batch_idx=batch_idx)
        self.log(
            "val_loss",
            loss,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch=batch, batch_idx=batch_idx)
        self.log(
            "test_loss",
            loss,
        )
        return loss

    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            factor=0.1,
            min_lr=1.0e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
