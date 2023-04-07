import numpy as np
import torch
from typing import Dict, Optional, List


class PredictorBundle:
    def __init__(self, model_dict: Dict):
        """Create a bundle of models based on a dictionary of models, all of them should take the same inputs

        Args:
            model_dict (Dict): dictionary with string keys and models as values
        """
        self.model_dict = model_dict

    def __call__(
        self, inputs: torch.tensor, model_subset_list: Optional[List[int]] = None
    ) -> torch.tensor:
        """Call all models sequentially, a subset of models can be used if ```models_subset_list''' is not None

        Args:
            inputs (torch.tensor): inputs (common for all models)
            model_subset_list (Optional[List[int]], optional):  subset of models to use. Defaults to None.

        Returns:
            torch.tensor: stacked results for all models
        """
        if model_subset_list is not None:
            keys = model_subset_list
        else:
            keys = list(self.model_dict.keys())

        output = np.stack([self.model_dict[k](inputs) for k in keys])
        return np.swapaxes(output, 0, 1)

    @property
    def s(
        self,
    ) -> np.array:
        """Get pair separation values for any of the models (they are all the same)

        Returns:
            np.array: array of pair separation values
        """
        return next(iter(self.model_dict.values())).s
