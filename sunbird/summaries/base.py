import numpy as np
from typing import Optional, List
import torch
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"

class BaseSummary:
    def __init__(
        self,
        path_to_model: Path,
        path_to_data: Optional[Path] =DEFAULT_DATA_PATH,
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
            "alpha_sat",
            "alpha_c",
            "sigma",
            "kappa",
        ]

    def __call__(
        self, param_dict, filters
    ):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]).reshape(1, -1),
            dtype=torch.float32,
        )
        output = self.get_for_sample(inputs, filters=filters)
        return output.reshape(-1)

    def get_for_sample(self, inputs, filters):
        return self.forward(inputs, filters=filters).detach().numpy()

    def get_for_batch(self, param_dict, filters,):
        inputs = torch.tensor(
            np.array([param_dict[k] for k in self.parameters]),
            dtype=torch.float32,
        ).T
        return self.get_for_batch_inputs(
            inputs=inputs, filters=filters,
        )

    def get_for_batch_inputs(self, inputs,filters):
        outputs = self.forward(inputs, filters=filters).detach().numpy()
        return outputs.reshape((len(inputs),-1))