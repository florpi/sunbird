import torch
from typing import Optional
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sunbird.models import Predictor, PredictorBundle
from sunbird.summaries.base import BaseSummary


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best_combined/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class DensitySplit(BaseSummary):
    def __init__(self, quintiles=[0,1,3,4], path_to_model=DEFAULT_PATH, path_to_data=DEFAULT_DATA_PATH, **kwargs):
        super().__init__(
            path_to_data=path_to_data,
            path_to_model=path_to_model,
        )
        self.quintiles = quintiles
        predictors_dict = {
            q: Predictor.from_folder(self.path_to_model / f'ds{q}/')
            for q in self.quintiles
        }
        self.model = PredictorBundle(
            predictors_dict,
        )

    def forward(self, inputs, filters):
        output = self.model(inputs)
        if filters is not None:
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
                output = output.reshape((len(output), -1, output.shape[-1]//2))
                output = output[:, filters['multipoles'], :].reshape(-1)
        return output
 