import numpy as np
import json
from pathlib import Path
from typing import List, Dict

# import xarray as xr
import torch
from matplotlib import pyplot as plt
from sunbird.summaries import DensitySplitAuto, DensitySplitCross, TPCF
from sunbird.abacus_utils.read_statistics import (
    read_statistics_for_covariance,
    read_statistic,
    read_parameters,
)

DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class CovarianceMatrix:
    def __init__(
        self,
        statistics: List[str],
        slice_filters: Dict,
        select_filters: Dict,
    ):
        """Compute a covariance matrix for a list of statistics and filters in any
        dimension

        Args:
            statistics (List[str]): list of statistics to use
            slice_filters (Dict): dictionary with slice filters on given coordinates
            select_filters (Dict): dictionary with select filters on given coordinates
        """
        self.statistics = statistics
        self.slice_filters = slice_filters
        self.select_filters = select_filters
        self.emulators = {
            'density_split_cross': DensitySplitCross(),
            'density_split_auto': DensitySplitAuto(),
            "tpcf": TPCF(),
        }

    def get_covariance_data(
        self,
    ) -> np.array:
        """Get the covariance matrix of the data for the specified summary statistics

        Returns:
            np.array: covariance matrix of the data
        """
        summaries = []
        for statistic in self.statistics:
            summary = read_statistics_for_covariance(
                statistic=statistic,
                select_filters=self.select_filters,
                slice_filters=self.slice_filters,
            )
            summary = np.array(summary.values).reshape((len(summary['phases']),-1))
            summaries.append(summary)
        summaries = np.hstack(summaries)
        return np.cov(summaries, rowvar=False)

    def get_true_test(
        self,
        test_cosmologies: List[int],
    ) -> np.array:
        """Get true values for the specified summary statistics in the test
        set cosmologies

        Args:
            test_cosmologies (List[int]): indices of test set cosmologies

        Returns:
            np.array: true values
        """
        xi_test = []
        for statistic in self.statistics:
            for cosmology in test_cosmologies:
                xi_test.append(
                    read_statistic(
                        statistic=statistic,
                        cosmology=cosmology,
                        dataset="different_hods",
                        select_filters=self.select_filters,
                        slice_filters=self.slice_filters,
                    ).values
                )
        return np.array(xi_test)

    def get_inputs_test(
        self,
        test_cosmologies: List[int],
    ) -> np.array:
        """Get input values for test set cosmologies

        Args:
            test_cosmologies (List[int]): indices of test set cosmologies

        Returns:
            np.array: input values
        """
        inputs = []
        for cosmology in test_cosmologies:
            inputs.append(
                read_parameters(
                    cosmology=cosmology, dataset="different_hods"
                ).to_numpy()
            )
        inputs = np.array(inputs)
        return inputs.reshape((-1, inputs.shape[-1]))

    def get_emulator_predictions(
        self,
        inputs: np.array,
    ) -> np.array:
        """Get emulator predictions for inputs

        Args:
            inputs (np.array): input data

        Returns:
            np.array: emulator prediction
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        xi_model = []
        for statistic in self.statistics:
            xi_model.append(
                self.emulators[statistic].get_for_batch_inputs(
                    inputs,
                    select_filters=self.select_filters,
                    slice_filters=self.slice_filters,
                ),
            )
        return np.squeeze(np.array(xi_model))

    def get_covariance_emulator_error(
        self,
    ) -> np.array:
        """Estimate the emulator's error on the test set

        Returns:
            np.array: covariance of the emulator's errors
        """
        with open(DATA_PATH / f"train_test_split.json", "r") as f:
            test_cosmologies = json.load(f)["test"]
        xi_test = self.get_true_test(test_cosmologies=test_cosmologies)
        inputs = self.get_inputs_test(test_cosmologies=test_cosmologies)
        xi_model = self.get_emulator_predictions(inputs=inputs)
        xi_test = xi_test.reshape(
            (xi_test.shape[0] * xi_test.shape[1], -1)
        )
        return np.cov(xi_model - xi_test, rowvar=False)


def normalize_cov(cov):
    nbins = len(cov)
    corr = np.zeros_like(cov)
    for i in range(nbins):
        for j in range(nbins):
            corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    return corr
