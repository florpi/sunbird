import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import torch
import json
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional


def load_summary_training(data_dir, statistic, s, apply_s2):
    with open(
        data_dir / f"train_{statistic}_summary.json",
        "r",
    ) as f:
        summary = json.load(f)
    for k, v in summary.items():
        if type(v) is list:
            summary[k] = np.array(v)
    if apply_s2:
        summary["y_min"] = summary["s2_y_min"]
        summary["y_max"] = summary["s2_y_max"]
        summary["y_mean"] *= s**2
        summary["y_std"] *= s**2
    return summary


class DSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        statistic: str = "ds0",
        batch_size: int = 32,
        standarize: bool = False,
        normalize: bool = False,
        normalize_inputs: bool = True,
        apply_s2: bool = False,
        s_min: Optional[float] = None,
        s_max: Optional[float] = None,
        dataset: Optional[str] = 'different_hods',
        #corr_type: Optional[str] = 'gaussian',
    ):
        super().__init__()
        self.data_dir= Path(__file__).parent.parent.parent / f"data/datasets/{dataset}/"
        self.statistic = statistic
        self.batch_size = batch_size
        self.standarize = standarize
        self.normalize = normalize
        self.normalize_inputs = normalize_inputs
        self.apply_s2 = apply_s2
        self.s = np.load(self.data_dir.parent.parent / "s.npy")
        self.s = np.array(list(self.s) + list(self.s))
        self.summary = load_summary_training(
            data_dir=self.data_dir,
            statistic=self.statistic,
            s=self.s,
            apply_s2=apply_s2,
        )
        self.s_min = s_min
        self.s_max = s_max

    def load_data(self, data_dir, stage):
        data = np.load(data_dir / f"{stage}_{self.statistic}.npy")
        if self.apply_s2:
            data = self.s**2 * data
        if self.normalize:
            data = (data - self.summary["y_min"]) / (
                self.summary["y_max"] - self.summary["y_min"]
            )
        elif self.standarize:
            data = (data - self.summary["y_mean"]) / self.summary["y_std"]
        if self.s_min is not None:
            mask_min = self.s > self.s_min
        else:
            mask_min = [True]*len(self.s)
        if self.s_max is not None:
            mask_max = self.s < self.s_max
        else:
            mask_max = [True]*len(self.s)
        if self.s_min is not None or self.s_max is not None:
            data = data[:, (mask_min) & (mask_max)]
        return data

    def load_parameters(self, data_dir, stage):
        parameters = np.load(data_dir / f"{stage}_params.npy")
        if self.normalize_inputs:
            parameters = (parameters - self.summary["x_min"]) / (
                self.summary["x_max"] - self.summary["x_min"]
            )
        return parameters

    def generate_dataset(self, x, y):
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def setup(self, stage: Optional[str] = None):
        if not stage or stage == "train" or stage == "fit":
            self.ds_train = self.generate_dataset(
                self.load_parameters(self.data_dir, stage="train"),
                self.load_data(self.data_dir, stage="train"),
            )
            self.n_output = self.ds_train.tensors[1].shape[-1]
            self.n_input = self.ds_train.tensors[0].shape[-1]

        if not stage or stage == "train" or stage == "fit":
            self.ds_val = self.generate_dataset(
                self.load_parameters(self.data_dir, stage="val"),
                self.load_data(self.data_dir, stage="val"),
            )
            self.n_output = self.ds_val.tensors[1].shape[-1]
            self.n_input = self.ds_val.tensors[0].shape[-1]

        if not stage or stage == "test":
            self.ds_test = self.generate_dataset(
                self.load_parameters(self.data_dir, stage="test"),
                self.load_data(self.data_dir, stage="test"),
            )
            self.n_output = self.ds_test.tensors[1].shape[-1]
            self.n_input = self.ds_test.tensors[0].shape[-1]

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

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

    def dump_summaries(self, path):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(path, 'w') as fd:
            json_dump = json.dumps(self.summary, cls=NumpyEncoder)
            json.dump(json_dump, fd)
