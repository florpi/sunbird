import torch
from typing import Optional, Dict
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
#from sunbird.models import Predictor, PredictorBundle
from sunbird.summaries.base import BaseSummary


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class DensitySplitBase(BaseSummary):
    def __init__(
        self,
        corr_type: str = "cross",
        path_to_model: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
    ):
        """Emulator for density split statistics

        Args:
            corr_type (str, optional): whether to use auto or cross correlations. Defaults to "cross".
            path_to_model (Path, optional):  path to where trained models are stored. Defaults to DEFAULT_PATH.
            path_to_data (Path, optional): path to data. Defaults to DEFAULT_DATA_PATH.
        """
        super().__init__(
            path_to_data=path_to_data,
            path_to_model=path_to_model,
        )
        predictors_dict = {
            q: Predictor.from_folder(self.path_to_model / f"ds_{corr_type}/ds{q}/", statistic='density_split')
            for q in [0, 1, 3, 4]
        }
        self.model = PredictorBundle(
            predictors_dict,
        )

    def forward(
        self, inputs: np.array, slice_filters: Dict = None, select_filters: Dict = None
    ) -> torch.tensor:
        """Generate density split statistics for a set of inputs. One can apply slice or select filters
        in any dimension of density split statitistics by using ```slice_filters''' or ```select_filters'''

        Args:
            inputs (np.array): inputs
            slice_filters (Dict, optional): dictionary with filters to slice across particular dimensions. Defaults to None.
            select_filters (Dict, optional): dictionary with filters to select across particular dimensions. Defaults to None.

        Returns:
            torch.tensor: density split statitiscs
        """
        output = self.model(
            inputs,
            model_subset_list=select_filters.get("quintiles", None)
            if select_filters is not None
            else None,
        )
        if slice_filters is not None:
            if "s" in slice_filters:
                s_min = slice_filters["s"][0]
                s_max = slice_filters["s"][1]
                output = output[..., (self.model.s > s_min) & (self.model.s < s_max)]
            if "multipoles" in slice_filters:
                m_min = slice_filters["multipoles"][0]
                m_max = slice_filters["multipoles"][1]
                output = output.reshape((output.shape[0], -1, 2, output.shape[-1] // 2))
                output = output[:, m_min:m_max, :]
        if select_filters is not None:
            if "multipoles" in select_filters:
                multipoles = select_filters["multipoles"]
                output = output.reshape((output.shape[0], -1, 2, output.shape[-1] // 2))
                output = output[..., multipoles, :]
        return output


class DensitySplitAuto(DensitySplitBase):
    def __init__(
        self,
    ):
        """Density split autocorrelations: autocorrelations between random points belonging to the same
        density quintile
        """
        super().__init__(
            corr_type="auto",
        )


class DensitySplitCross(DensitySplitBase):
    def __init__(
        self,
    ):
        """Density split cross-correlations: cross corrrelations between random points belonging to the same
        density quintile and the full galaxy field
        """

        super().__init__(
            corr_type="cross",
        )
