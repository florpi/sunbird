from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import jax.numpy as jnp
from sunbird.summaries.base import BaseSummary
from sunbird.summaries import TPCF, DensitySplitAuto, DensitySplitCross, DensityPDF


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"


class Bundle(BaseSummary):
    def __init__(
        self,
        summaries: List[str],
        dataset: str = "bossprior",
        n_hod_realizations: Optional[int] = None,
        suffix: Optional[str] = None,
        loss: str = "mae",
        flax: bool = False,
        path_to_models: Path = DEFAULT_PATH,
    ):
        """Combine a list of summaries into a bundle

        Args:
            summaries (List[str]): list of summaries to combine
        """
        self.summaries = summaries
        self.flax = flax 
        self.all_summaries = {
            "tpcf": TPCF(
                dataset=dataset,
                loss=loss,
                flax=flax,
                n_hod_realizations=n_hod_realizations,
                path_to_models=path_to_models,
                suffix=suffix,
            ),
            "density_split_cross": DensitySplitCross(
                dataset=dataset,
                loss=loss,
                flax=flax,
                n_hod_realizations=n_hod_realizations,
                suffix=suffix,
                path_to_models=path_to_models,
            ),
            "density_split_auto": DensitySplitAuto(
                dataset=dataset,
                loss=loss,
                flax=flax,
                n_hod_realizations=n_hod_realizations,
                suffix=suffix,
                path_to_models=path_to_models,
            ),
            "density_pdf": DensityPDF(
                dataset=dataset,
                loss=loss,
                flax=flax,
                n_hod_realizations=n_hod_realizations,
                suffix=suffix,
                path_to_models=path_to_models,
            ),
        }

    @property
    def input_names(
        self,
    ):
        return self.all_summaries["density_split_auto"].input_names

    def forward(
        self,
        inputs: np.array,
        select_filters: Dict = None,
        slice_filters: Dict = None,
        batch: bool = False,
        use_xarray: bool = False,
    ) -> np.array:
        """return a concatenated prediction of all the summaries

        Args:
            inputs (np.array): input values to predict for.
            select_filters (Dict): filters to select values in a given dimension.
            slice_filters (Dict): filters to select values within slice in a given dimension.

        Returns:
            np.array: emulator predictions
        """
        output, output_uncertainty = [], []
        for summary in self.summaries:
            pred, pred_uncertainty = self.all_summaries[summary].forward(
                inputs=inputs,
                select_filters=select_filters,
                slice_filters=slice_filters,
                batch=batch,
                use_xarray=use_xarray,
            )
            if batch and not use_xarray:
                pred = pred.reshape((len(inputs), -1))
                pred_uncertainty = pred_uncertainty.reshape((len(inputs), -1))
            output.append(pred)
            output_uncertainty.append(pred_uncertainty)
        if use_xarray:
            return output, output_uncertainty
        if self.flax:
            return jnp.hstack(output).squeeze(), jnp.hstack(output_uncertainty).squeeze()
        return np.hstack(output).squeeze(), np.hstack(output_uncertainty).squeeze()
