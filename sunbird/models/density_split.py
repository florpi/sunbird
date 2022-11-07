import torch
from typing import Optional
import numpy as np
from pathlib import Path
from sunbird.models import FCN
import matplotlib.pyplot as plt


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class DensitySplit:
    def __init__(self, path_to_models=DEFAULT_PATH, path_to_data=DEFAULT_DATA_PATH):
        path_to_models = Path(path_to_models)
        self.s = np.load(path_to_data / "s.npy")
        self.quintiles = [0, 1, 3, 4]
        self.models = {
            q: FCN.from_folder(path_to_models / f"best/ds{q}_m0/")
            for q in self.quintiles
        }
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
            "alpha_sat",
            "alpha_c",
            "sigma",
            "kappa",
        ]

    def __call__(
        self,
        param_dict,
        quintiles,
        s_min = None,
    ):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]).reshape(1, -1),
            dtype=torch.float32,
        )
        output = np.vstack(
            [self.models[q](inputs)[0].detach().numpy() for q in quintiles]
        )
        if s_min is not None:
            return output[:, self.s > s_min].reshape(-1)
        return output.reshape(-1)

    def get_for_sample(self, inputs):
        return np.vstack(
            [self.models[q](inputs)[0].detach().numpy() for q in self.quintiles]
        )

    def get_for_batch(self, param_dict, quintiles, s_min: Optional[float] = None):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]),
            dtype=torch.float32,
        ).T
        return self.get_for_batch_inputs(
            inputs=inputs, s_min=s_min, quintiles=quintiles
        )

    def get_for_batch_inputs(self, inputs, quintiles, s_min: Optional[float] = None):
        outputs = np.vstack(
            [self.models[q](inputs).detach().numpy() for q in quintiles],
        )
        if s_min is not None:
            return outputs[:, self.s > s_min]
        return outputs
