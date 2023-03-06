import numpy as np
from typing import Optional, List
import torch
from pathlib import Path
from matplotlib import pyplot as plt

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class BaseSummary:
    def __init__(
        self,
        path_to_model: Path,
        path_to_data: Optional[Path] = DEFAULT_DATA_PATH,
    ):
        self.path_to_model = Path(path_to_model)
        self.path_to_data = Path(path_to_data)
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

    def __call__(self, param_dict, select_filters, slice_filters):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]).reshape(1, -1),
            dtype=torch.float32,
        )
        output = self.get_for_sample(
            inputs, select_filters=select_filters, slice_filters=slice_filters
        )
        return output.reshape(-1)

    def get_for_sample(self, inputs, select_filters, slice_filters):
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
