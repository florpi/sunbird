import torch
from typing import Optional
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sunbird.models import Predictor, PredictorBundle
from sunbird.summaries.base import BaseSummary


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best_gaussian/"
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

    def forward(self, inputs, slice_filters, select_filters):
        output = self.model(inputs)
        if slice_filters is not None:
            if 's' in slice_filters:
                s_min = slice_filters['s'][0]
                s_max = slice_filters['s'][1]
                output = output[:, (self.model.s > s_min) & (self.model.s < s_max)]
            if 'multipoles' in slice_filters:
                m_min = slice_filters['multipoles'][0]
                m_max = slice_filters['multipoles'][1]
                output = output.reshape((len(output), -1, output.shape[-1]//2))
                output = output[:, m_min:m_max, :].reshape(-1)
        if select_filters is not None:
            if 'multipoles' in select_filters:
                multipoles = select_filters['multipoles']
                output = output.reshape((len(output), -1, output.shape[-1]//2))
                output = output[:, multipoles, :].reshape(-1)
        return output
        
