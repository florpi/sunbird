import torch
import numpy as np
from typing import Dict
from pathlib import Path
import yaml
import lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from flax.traverse_util import unflatten_dict

import sunbird.emulators.models as models


def convert_state_dict_from_pt(
    model,
    state,
):
    """
    Converts a PyTorch parameter state dict to an equivalent Flax parameter state dict
    """
    state = {k: v.numpy() for k, v in state.items()}
    state = model.convert_from_pytorch(
        state,
    )
    state = unflatten_dict({tuple(k.split(".")): v for k, v in state.items()})
    return state


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """Base pytorch lightning model"""
        super().__init__()

    @classmethod
    def from_folder(
        cls,
        path_to_model: str,
        load_loss: bool = False,
    ) -> "BaseModel":
        """load a model from folder

        Args:
            path_to_model (str): path to model folder

        Returns:
            model: model loaded from checkpoint
        """
        path_to_model = Path(path_to_model)
        with open(path_to_model / "hparams.yaml") as f:
            hparams = yaml.safe_load(f)
            del hparams["load_loss"]
        model = cls(**hparams, load_loss=False)
        # find file with lowest validation loss
        files = list((path_to_model / "checkpoints").glob("*.ckpt"))
        file_idx = np.argmin(
            [float(str(file).split(".ckpt")[0].split("=")[-1]) for file in files]
        )
        weights_dict = torch.load(
            files[file_idx],
            map_location=torch.device("cpu"),
        )
        state_dict = weights_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return model

    def training_step(self, batch, batch_idx) -> float:
        """Compute training loss

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
            logger=False,
        )
        #print(f'Epoch = {self.current_epoch}, step = {self.trainer.global_step}, wandb step = {self.logger.experiment.step}')
        # self.logger.experiment.log({'train_loss': loss.item(), 'global_step': self.trainer.global_step+1}, step=self.trainer.global_step+1)
        # self.log(
        #     "train_loss",
        #     loss.item(),
        #     prog_bar=True,
        #     logger=False,
        # )
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        """Compute validation loss

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
            logger=False,
        )
        # self.logger.experiment.log({'val_loss': loss.item(), 'global_step': self.trainer.global_step}, step=self.trainer.global_step)
        # self.log(
        #     "val_loss",
        #     loss,
        #     prog_bar=True,
        #     logger=False,
        # )
        return loss

    def test_step(self, batch, batch_idx):
        """Compute test loss

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

    def configure_optimizers(self) -> Dict:
        """configure optimizer and learning rate scheduler

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
            patience=self.scheduler_patience, #50,
            factor=self.scheduler_factor, #0.5,
            threshold=self.scheduler_threshold,
            threshold_mode='abs',
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
    
    @property
    def flax_attributes(self,):
        return

    def to_jax(self,):
        nn_model = getattr(models, f'Flax{self.__class__.__name__}')(
            **self.flax_attributes,

        ) 
        flax_params = {'params': convert_state_dict_from_pt(
            model=nn_model,
            state=self.state_dict(),
        )}
        return nn_model, flax_params