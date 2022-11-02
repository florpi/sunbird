import torch
from typing import Optional
import numpy as np
from pathlib import Path
from sunbird.models import FCN
import matplotlib.pyplot as plt


DEFAULT_PATH = Path("/n/home11/ccuestalazaro/sunbird/trained_models/")
DEFAULT_DATA_PATH = Path("/n/home11/ccuestalazaro/sunbird/data/full_ap/")


class DensitySplit:
    def __init__(self, path_to_models=DEFAULT_PATH, path_to_data=DEFAULT_DATA_PATH):
        path_to_models = Path(path_to_models)
        self.s = np.load(path_to_data / "s.npy")
        self.best_models = {
            0: 121,
            1: 142,
            3: 143,
            4: 185,
        }
        self.models = [
            FCN.from_folder(
                path_to_models / f"optuna_ds{q}_m0/version_{self.best_models[q]}"
            )
            for q in self.best_models.keys()
        ]
        self.parameters = [
            "omega_b",
            "omega_cdm",
            "sigma8_m",
            "n_s",
            "alpha_s",
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
    ):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]).reshape(1, -1),
            dtype=torch.float32,
        )
        return np.vstack([model(inputs)[0].detach().numpy() for model in self.models])

    def get_for_sample(self, inputs):
        return np.vstack([model(inputs)[0].detach().numpy() for model in self.models])

    def get_for_batch(self, param_dict, s_min: Optional[float] = None):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]),
            dtype=torch.float32,
        ).T
        return self.get_for_batch_inputs(inputs=inputs, s_min=s_min)

    def get_for_batch_inputs(self, inputs, s_min: Optional[float] = None):
        outputs = np.vstack([model(inputs).detach().numpy() for model in self.models])
        if s_min is not None:
            return outputs[:, self.s > s_min]
        return outputs
