import torch
import numpy as np
from typing import Dict
from pathlib import Path
import yaml
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau 


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """ Base pytorch lightning model
        """
        super().__init__()

    @classmethod
    def from_folder(cls, path_to_model: str)->"BaseModel":
        """ load a model from folder

        Args:
            path_to_model (str): path to model folder

        Returns:
            model: model loaded from checkpoint 
        """
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
        state_dict = {k: v for k, v in weights_dict['state_dict'].items() if k != 'covariance'}
        model.load_state_dict(state_dict, strict=False)
        return model

    def training_step(self, batch, batch_idx)->float:
        """ Compute training loss

        Args:
            batch: batch 
            batch_idx: idx of batch 

        Returns:
            float: loss 
        """
        loss = self._compute_loss(batch=batch, batch_idx=batch_idx)
        self.log(
            "train_loss",
            loss,
        )
        return loss

    def validation_step(self, batch, batch_idx)->float:
        """ Compute validation loss

        Args:
            batch: batch 
            batch_idx: idx of batch 

        Returns:
            float: loss 
        """
        loss = self._compute_loss(batch=batch, batch_idx=batch_idx)
        self.log(
            "val_loss",
            loss,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """ Compute test loss

        Args:
            batch: batch 
            batch_idx: idx of batch 

        Returns:
            float: loss 
        """
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

    def configure_optimizers(self)->Dict:
        """ configure optimizer and learning rate scheduler

        Returns:
            Dict: dictionary with configuration 
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
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
