from pathlib import Path
import torch
import json
import numpy as np
from sunbird.models.predictor import Predictor
from sunbird.summaries.base import BaseSummary

DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/tpcf/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"
#TODO: Take care of normalization properly, by reading from hparams.yaml

class TPCF(BaseSummary):
    def __init__(self, path_to_model: Path = DEFAULT_PATH, path_to_data:Path=DEFAULT_DATA_PATH, **kwargs):
        super().__init__(
            path_to_data=path_to_data,
            path_to_model=path_to_model,
        )
        self.model = Predictor.from_folder(path_to_model)

    def forward(self, inputs, filters):
        output = self.model(inputs)
        if 's_min' in filters:
            min_mask = self.model.s > filters['s_min']
        else:
            min_mask = None
        if 's_max' in filters:
            max_mask = self.model.s < filters['s_max']
        else:
            max_mask = None
        output = output[:, (min_mask) & (max_mask)]
        if 'multipoles' in filters:
            output = output.reshape((-1, output.shape[-1]//2))[filters['multipoles']].reshape(-1)
        return output
        