from typing import List, Dict
import torch
from sunbird.summaries.base import BaseSummary


class Bundle(BaseSummary):
    def __init__(
        self,
        summaries: List["BaseSummary"],
    ):
        """Combine a list of summaries into a bundle

        Args:
            summaries (List[&quot;BaseSummary&quot;]): list of summaries
        """

        self.summaries = summaries

    def forward(
        self, inputs: torch.tensor, select_filters: Dict, slice_filters: Dict
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
                summary.forward(
                    inputs=inputs,
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                ).reshape((len(inputs), -1))
            )
        return torch.hstack(output)
