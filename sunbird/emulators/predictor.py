import torch
import numpy as np
import yaml
import json
from typing import List
from pathlib import Path
import pytorch_lightning as pl
import xarray as xr
#from sunbird.data.data_utils import inverse_transform_summary
from sunbird.emulators import FCN

TRAIN_DIR = Path(__file__).parent


class Predictor(pl.LightningModule):
    def __init__(
        self,
        nn_model,
        summary_stats,
        statistic: str,
        normalize_inputs: bool = True,
        normalize_outputs: bool = False,
        standarize_outputs: bool = False,
    ):
        super().__init__()
        self.nn_model = nn_model
        self.statistic=statistic
        self.summary_stats = summary_stats
        self.s = self.summary_stats['coordinates']['s']
        self.normalize_outputs = normalize_outputs
        self.standarize_outputs = standarize_outputs
        self.normalize_inputs = normalize_inputs

    @classmethod
    def from_folder(cls, path_to_model: Path, statistic:str):
        path_to_model = Path(path_to_model)
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        nn_model = FCN.from_folder(path_to_model)
        nn_model.eval()
        summary_stats = cls.load_summary(path_to_model)
        for k, v in summary_stats.items():
            if k == 'coordinates':
                del summary_stats['coordinates']['quintiles']
                del summary_stats['coordinates']['realizations']
            else:
                summary_stats[k] = torch.tensor(v, dtype=torch.float32, requires_grad=False)
        return cls(
            nn_model=nn_model,
            statistic=statistic,
            summary_stats=summary_stats,
            normalize_inputs=config["normalize_inputs"],
            normalize_outputs=config["normalize_outputs"],
            standarize_outputs=config["standarize_outputs"],
        )

    @classmethod
    def load_summary(cls, path_to_model):
        with open(path_to_model / "summary.json") as fd:
            summary = json.load(fd)
        return eval(summary)

    def forward(self, inputs: torch.tensor, return_xarray: bool = False,):
        if self.normalize_inputs:
            inputs = (inputs - self.summary_stats["x_min"]) / (
                self.summary_stats["x_max"] - self.summary_stats["x_min"]
            )
        prediction = self.nn_model(inputs)
        if self.normalize_outputs:
            prediction = (
                prediction * (self.summary_stats["y_max"] - self.summary_stats["y_min"])
                + self.summary_stats["y_min"]
            )
        elif self.standarize_outputs:
            prediction = (
                prediction * self.summary_stats["y_std"] + self.summary_stats["y_mean"]
            )
        # Undo transforms
        self.summary_stats['coordinates']['realizations'] = range(len(prediction))
        prediction = xr.DataArray(
            prediction.detach().numpy().reshape((len(prediction), -1, len(self.s))),
            dims =['realizations', 'multipoles', 's'],
            coords=self.summary_stats['coordinates'],
        )
        prediction = inverse_transform_summary(prediction, statistic=self.statistic)
        if return_xarray:
            return prediction
        return prediction.values
