from typing import List, Dict
import numpy as np
import torch
from sunbird.summaries.base import BaseSummary
from sunbird.summaries import TPCF, DensitySplitAuto, DensitySplitCross


class Bundle(BaseSummary):
    def __init__(
        self,
        summaries: List[str],
    ):
        """Combine a list of summaries into a bundle

        Args:
            summaries (List[str]): list of summaries to combine
        """
        self.summaries = summaries
        self.all_summaries = {
            #'tpcf': TPCF(),
            'density_split_cross': DensitySplitCross(),
            'density_split_auto': DensitySplitAuto(),
        }
        
    @property
    def parameters(self,):
        return self.all_summaries['density_split_auto'].parameters

    def forward(
        self, inputs: torch.tensor, select_filters: Dict=None, slice_filters: Dict=None,
    ) -> torch.tensor:
        """return a concatenated prediction of all the summaries

        Args:
            inputs (torch.tensor): input values to predict for .
            select_filters (Dict): filters to select values in a given dimension.
            slice_filters (Dict): filters to select values within slice in a given dimension.

        Returns:
            torch.tensor: emulator predictions
        """
        output = []
        for summary in self.summaries:
            output.append(
                self.all_summaries[summary].forward(
                    inputs=inputs,
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                ).reshape((len(inputs), -1))
            )
        return np.hstack(output)
