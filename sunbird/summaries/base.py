import numpy as np
import json
import yaml
from typing import List
import torch
from pathlib import Path

from sunbird.emulators import FCN, FlaxFCN
from flax.core.frozen_dict import freeze

from flax.traverse_util import  unflatten_dict


DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"

def convert_state_dict_from_pt(
    model, state,
):
    """
    Converts a PyTorch parameter state dict to an equivalent Flax parameter state dict
    """
    state = {k: v.numpy() for k, v in state.items()}
    state = model.convert_from_pytorch(state,)
    state = unflatten_dict({tuple(k.split(".")): v for k, v in state.items()})
    return state


class BaseSummary:
    def __init__(
        self,
        model,
        summary_stats,
        normalize_inputs: bool = True, 
        normalize_outputs: bool=True,
        standarize_outputs: bool = False,
        flax_params = None
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
        self.parameters = [
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
        ]

    @classmethod
    def from_folder(cls, path_to_model: Path, flax: bool = False,):
        path_to_model = Path(path_to_model)
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        if flax:
            nn_model = FlaxFCN(
                n_input = config['n_input'],
                n_hidden= config['n_hidden'],
                act_fn= config['act_fn'],
                n_output= config['n_output'],
            )
            files = list((path_to_model / "checkpoints").glob("*.ckpt"))
            file_idx = np.argmin(
                [float(str(file).split(".ckpt")[0].split("=")[-1]) for file in files]
            )
            weights_dict = torch.load(
                files[file_idx],
                map_location=torch.device("cpu"),
            )
            state_dict = weights_dict['state_dict']
            flax_params = convert_state_dict_from_pt(model=nn_model, state=state_dict,)
        else:
            nn_model = FCN.from_folder(path_to_model, load_loss=False,)
            nn_model.eval()
            flax_params = None
        summary_stats = cls.load_summary(path_to_model)
        for k, v in summary_stats.items():
            if flax:
                summary_stats[k] = np.array(v)
            else:
                summary_stats[k] = torch.tensor(v, dtype=torch.float32, requires_grad=False)
        return cls(
            model=nn_model,
            summary_stats=summary_stats,
            normalize_inputs=config["normalize_inputs"],
            normalize_outputs=config["normalize_outputs"],
            standarize_outputs=config["standarize_outputs"],
            flax_params=flax_params,
        )


    @classmethod
    def load_summary(cls, path_to_model):
        with open(path_to_model / "summary.json") as fd:
            summary = json.load(fd)
        return eval(summary)


    def forward(self, inputs: torch.tensor, select_filters, slice_filters, return_xarray: bool = False,):
        if self.normalize_inputs:
            inputs = (inputs - self.summary_stats["x_min"]) / (
                self.summary_stats["x_max"] - self.summary_stats["x_min"]
            )
        if self.flax:
            prediction = self.model.apply(freeze({"params": self.flax_params}), inputs)
        else:
            prediction = self.model(inputs)
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
        #self.summary_stats['coordinates']['realizations'] = range(len(prediction))
        return prediction
        #prediction = xr.DataArray(
        #    prediction.detach().numpy().reshape((len(prediction), -1, len(self.s))),
        #    dims =['realizations', 'multipoles', 's'],
        #    coords=self.summary_stats['coordinates'],
        #)
        #prediction = inverse_transform_summary(prediction, statistic=self.statistic)
        #if return_xarray:
        #    return prediction
        #return prediction.values

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
            return (
                self.forward(
                    inputs, select_filters=select_filters, slice_filters=slice_filters
                )
            )
        else:
            return (
                self.forward(
                    inputs, select_filters=select_filters, slice_filters=slice_filters
                )
                .detach()
                .numpy()
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
