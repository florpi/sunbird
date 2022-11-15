import numpy as np
import torch
from typing import Dict

class PredictorBundle:
    def __init__(self, model_dict: Dict):
        self.model_dict = model_dict

    def __call__(self, inputs: torch.tensor):
        return torch.vstack(
            [model(inputs) for k, model in self.model_dict.items()]
        )

    @property
    def s(self,):
        return next(iter(self.model_dict.values())).s
