import numpy as np
import jax.numpy as jnp
import json
import yaml
from typing import List, Optional
import torch
from pathlib import Path

from sunbird.emulators import FCN, FlaxFCN
from sunbird.data import transforms 
from sunbird.data.data_utils import convert_selection_to_filters, convert_to_summary
from flax.core.frozen_dict import freeze


DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class BaseSummary:
    def __init__(
        self,
        model,
        summary_stats,
        coordinates,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
        standarize_outputs: bool = False,
        transform:  Optional[str] = None,
        flax_params: Optional[jnp.array] =None,
        parameters: Optional[List] = [
            "omega_b",
            "omega_cdm",
            "sigma8_m",
            "n_s",
            "nrun",
            "N_ur",
            "w0_fld",
            "wa_fld",
            "logM1",
            "logM_cut",
            "alpha",
            "alpha_s",
            "alpha_c",
            "logsigma",
            "kappa",
            "B_cen",
            "B_sat",
        ],
    ):
        self.model = model
        self.flax_params = flax_params
        if self.flax_params is not None:
            self.flax = True
        else:
            self.flax = False
        self.summary_stats = summary_stats
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.standarize_outputs = standarize_outputs
        if transform is not None:
            self.transform = getattr(transforms, transform)(
                normalization_dict = summary_stats,
            )
        else:
            self.transform = None
        self.parameters = parameters
        self.coordinates = coordinates
        self.coordinates_shape = tuple(len(v) for k, v in coordinates.items())

    @classmethod
    def from_folder(
        cls,
        path_to_model: Path,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
    ):
        path_to_model = Path(path_to_model)
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        if flax:
            nn_model, flax_params = FlaxFCN.from_folder(
                path_to_model,
            )
        else:
            nn_model = FCN.from_folder(
                path_to_model,
                load_loss=False,
            )
            nn_model.eval()
            flax_params = None
        summary_stats = cls.load_summary(path_to_model)
        for k, v in summary_stats.items():
            if flax:
                summary_stats[k] = np.array(v)
            else:
                summary_stats[k] = torch.tensor(
                    v, dtype=torch.float32, requires_grad=False
                )
        coordinates = cls.load_coordinates(config=config, path_to_data=path_to_data)
        return cls(
            model=nn_model,
            summary_stats=summary_stats,
            coordinates=coordinates,
            normalize_inputs=config["normalize_inputs"],
            normalize_outputs=config["normalize_outputs"],
            standarize_outputs=config["standarize_outputs"],
            flax_params=flax_params,
        )

    @classmethod
    def load_coordinates(cls, config, path_to_data):
        with open(path_to_data / f'coordinates/{config["statistic"]}.json') as fd:
            coordinates = json.load(fd)
        coordinates = {k: np.array(v) for k, v in coordinates.items()}
        select_filters, slice_filters = convert_selection_to_filters(config)
        for key, values in select_filters.items():
            coordinates[key] = [v for v in coordinates[key] if v in values]
        for key, values in slice_filters.items():
            min_value, max_value = slice_filters[key]
            coordinates[key] = coordinates[key][(coordinates[key] > min_value) & (coordinates[key] < max_value)]
        return coordinates

    @classmethod
    def load_summary(cls, path_to_model):
        with open(path_to_model / "summary.json") as fd:
            summary = json.load(fd)
        return eval(summary)

    def invert_transforms(self, prediction):
        if self.normalize_outputs:
            prediction = (
                prediction * (self.summary_stats["y_max"] - self.summary_stats["y_min"])
                + self.summary_stats["y_min"]
            )
        elif self.standarize_outputs:
            prediction = (
                prediction * self.summary_stats["y_std"] + self.summary_stats["y_mean"]
            )
        if self.transform is not None:
            prediction = self.transform.inverse_transform(prediction)
        return prediction

    def forward(
        self,
        inputs: torch.tensor,
        select_filters,
        slice_filters,
        return_xarray: bool = False,
    ):
        if self.normalize_inputs:
            inputs = (inputs - self.summary_stats["x_min"]) / (
                self.summary_stats["x_max"] - self.summary_stats["x_min"]
            )
        if self.flax:
            prediction = self.model.apply(freeze({"params": self.flax_params}), inputs)
        else:
            prediction = self.model(inputs)
        # Invert transform
        prediction = self.apply_filters(
            prediction=prediction,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        prediction = self.invert_transforms(prediction)
        if return_xarray:
            return prediction
        return prediction.values.reshape(-1)

    def apply_filters(self, prediction, select_filters, slice_filters):
        data = prediction.reshape(self.coordinates_shape)
        if not self.flax:
            data = data.detach().numpy()
        return convert_to_summary(
            data=data,
            dimensions=list(self.coordinates.keys()),
            coords=self.coordinates,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )

    def __call__(self, param_dict, select_filters=None, slice_filters=None):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]).reshape(1, -1),
            dtype=torch.float32,
        )
        output = self.get_for_sample(
            inputs, select_filters=select_filters, slice_filters=slice_filters
        )
        return output.reshape(-1)

    def get_for_sample(self, inputs, select_filters, slice_filters):
        if self.flax:
            return self.forward(
                inputs, select_filters=select_filters, slice_filters=slice_filters
            )
        else:
            return (
                self.forward(
                    inputs, select_filters=select_filters, slice_filters=slice_filters
                )
                #.detach()
                #.numpy()
            )

    def get_for_batch(
        self,
        param_dict,
        select_filters,
        slice_filters,
    ):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]),
            dtype=torch.float32,
        ).T
        return self.get_for_batch_inputs(
            inputs=inputs,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )

    def get_for_batch_inputs(
        self,
        inputs,
        select_filters,
        slice_filters,
    ):
        outputs = (
            self.forward(
                inputs,
                select_filters=select_filters,
                slice_filters=slice_filters,
            )
            .detach()
            .numpy()
        )
        return outputs.reshape((len(inputs), -1))
