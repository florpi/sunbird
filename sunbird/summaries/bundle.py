from typing import List, Dict
import time
import numpy as np
from sunbird.summaries.base import BaseSummary
from sunbird.summaries import TPCF, DensitySplitAuto, DensitySplitCross

class Bundle(BaseSummary):
    def __init__(
        self,
        summaries: List[str],
        dataset: str = 'boss_wideprior',
        loss: str = 'mae',
        flax: bool = False,
    ):
        """Combine a list of summaries into a bundle

        Args:
            summaries (List[str]): list of summaries to combine
        """
        self.summaries = summaries
        self.flax = False
        self.all_summaries = {
            #'tpcf': TPCF,
            'density_split_cross': DensitySplitCross(dataset=dataset, loss=loss,flax=flax,),
            'density_split_auto': DensitySplitAuto(dataset=dataset, loss=loss,flax=flax,),
        }
        
    @property
    def input_names(self,):
        return self.all_summaries['density_split_auto'].input_names

    def forward(
        self, inputs: np.array, select_filters: Dict=None, slice_filters: Dict=None, batch: bool = False,
    ) -> np.array:
        """return a concatenated prediction of all the summaries

        Args:
            inputs (np.array): input values to predict for.
            select_filters (Dict): filters to select values in a given dimension.
            slice_filters (Dict): filters to select values within slice in a given dimension.

        Returns:
            np.array: emulator predictions
        """
        output = []
        for summary in self.summaries:
            output.append(
                self.all_summaries[summary].forward(
                    inputs=inputs,
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                    batch=batch,
                ).reshape((len(inputs), -1))
            )
        return np.hstack(output)
