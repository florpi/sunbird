import numpy as np
import json
from pathlib import Path
from scipy.stats import sigmaclip
from typing import List, Dict, Optional

import torch
from sunbird.read_utils import data_utils

DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class CovarianceMatrix:
    def __init__(
        self,
        statistics: List[str],
        slice_filters: Dict,
        select_filters: Dict,
        covariance_data_class: str = 'AbacusSmall',
        emulator_data_class: str = 'Abacus',
        standarize_covariance: bool = False,
        normalize_covariance: bool = False,
        normalization_dict: Optional[Dict] = None
    ):
        """Compute a covariance matrix for a list of statistics and filters in any
        dimension

        Args:
            statistics (List[str]): list of statistics to use
            slice_filters (Dict): dictionary with slice filters on given coordinates
            select_filters (Dict): dictionary with select filters on given coordinates
        """
        self.covariance_data = getattr(data_utils, covariance_data_class)(
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
            standarize=standarize_covariance,
            normalize=normalize_covariance,
            normalization_dict=normalization_dict,
        )
        self.covariance_simulation = getattr(data_utils, "AbacusSmall")(
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
            standarize=standarize_covariance,
            normalize=normalize_covariance,
            normalization_dict=normalization_dict,
        )
        self.emulator_data = getattr(data_utils, emulator_data_class)(
            dataset="wideprior_AB",
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
        )
        self.covariance_data_class = covariance_data_class
        self.statistics = statistics
        self.slice_filters = slice_filters
        self.select_filters = select_filters
        self.standarize_covariance = standarize_covariance
        self.normalize_covariance = normalize_covariance
        self.normalization_dict = normalization_dict

    def get_covariance_data(
        self,
        volume_scaling: float = None,
        apply_hartlap_correction: bool = True,
        fractional: bool = False,
    ) -> np.array:
        """Get the covariance matrix of the data for the specified summary statistics

        Args:
            volume_scaling (float): volume scaling factor
            apply_hartlap_correction (bool, optional): whether to apply hartlap correction. Defaults to True.
            fractional (bool, optional): whether to return the fractional covariance matrix. Defaults to False.

        Returns:
            np.array: covariance matrix of the data
        """
        if volume_scaling is None:
            if self.covariance_data_class == 'AbacusSmall':
                raise ValueError("Volume scaling must be specified when using AbacusSmall covariance class.")
            else:
                volume_scaling = 1.0
        return self.covariance_data.get_covariance(
            apply_hartlap_correction=apply_hartlap_correction,
            fractional=fractional,
            volume_scaling=volume_scaling,
        )

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
        xi_tests = []
        for statistic in self.statistics:
            xi_test = []
            for cosmology in test_cosmologies:
                xi = self.emulator_data.read_statistic(
                        statistic=statistic,
                        cosmology=cosmology,
                        phase=0,
                    ).values
                xi_test.append(xi.reshape(xi.shape[0], -1))
            xi_test = np.asarray(xi_test)
            xi_tests.append(xi_test.reshape(xi_test.shape[0] * xi_test.shape[1], -1))
        return np.concatenate(xi_tests, axis=-1)

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
                self.emulator_data.get_all_parameters(
                    cosmology=cosmology, 
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
        if not hasattr(self, "emulators"):
            from sunbird.summaries import DensitySplitAuto, DensitySplitCross, TPCF
            self.emulators = {
                'density_split_cross': DensitySplitCross(),
                'density_split_auto': DensitySplitAuto(),
                "tpcf": TPCF(),
            }
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
        xi_model = np.hstack(xi_model)
        return np.squeeze(np.array(xi_model))
    
    def get_covariance_simulation(
        self,
        apply_hartlap_correction: bool = True,
        fractional: bool = False,
    ) -> np.array:
        """Get the covariance matrix associated with the finite volume
        of the simulations used to train the emulator.

        Args:
            apply_hartlap_correction (bool, optional): whether to apply hartlap correction. Defaults to True.
            fractional (bool, optional): whether to return the fractional covariance matrix. Defaults to False.

        Returns:
            np.array: covariance matrix of the simulations sample variance.
        """
        return self.covariance_simulation.get_covariance(
            apply_hartlap_correction=apply_hartlap_correction,
            fractional=fractional,
            volume_scaling=64,
        )

    def get_covariance_emulator(
        self,
        xi_data: np.array = None,
        covariance_data: np.array = None,
        fractional: bool = False,
        clip_errors: bool = True,
        clipping_factor: float = 3.0,
        return_mean: bool = False,

    ) -> np.array:
        """Estimate the emulator's error on the test set

        Args:
            xi_data (np.array): observed data vector
            covariance_data (np.array): covariance matrix of the data
            fractional (bool, optional): whether to return the fractional covariance matrix. Defaults to False.
            clip_errors (bool, optional): whether to clip the errors. Defaults to False.
            clipping_factor (float, optional): clipping factor. Defaults to 3.0.
            return_mean (bool, optional): whether to return the mean of the clipped errors. Defaults to False.

        Returns:
            np.array: covariance of the emulator's errors
        """
        with open(DATA_PATH / f"train_test_split.json", "r") as f:
            test_cosmologies = json.load(f)["test"]
        xi_test = self.get_true_test(test_cosmologies=test_cosmologies)
        inputs = self.get_inputs_test(test_cosmologies=test_cosmologies)
        xi_model = self.get_emulator_predictions(inputs=inputs)
        if clip_errors:
            if covariance_data is None or xi_data is None:
                raise ValueError("Covariance data and xi_data must be specified when clipping errors.")
            xi_test, mask = self.clip_xi_test(
                xi_test=xi_test,
                xi_data=xi_data,
                covariance_data=covariance_data,
                return_mask=True,
                clipping_factor=clipping_factor,
            )
            xi_model = xi_model[mask]
        absolute_error = xi_model - xi_test
        if fractional:
            if return_mean:
                return np.cov(absolute_error/xi_test, rowvar=False), np.mean(absolute_error/xi_test, axis=0)
            return np.cov(absolute_error/xi_test, rowvar=False)
        if return_mean:
            return np.cov(absolute_error, rowvar=False), np.mean(absolute_error, axis=0)
        return np.cov(absolute_error, rowvar=False)

    def clip_xi_test(
        self,
        xi_test: np.array,
        xi_data: np.array,
        covariance_data: np.array,
        clipping_factor: float = 3.0,
        return_mask: bool = False,
    ) -> np.array:
        """Clip data to remove outliers

        Args:
            xi_test (np.array): test set to clip
            xi_data (np.array): observed data vector
            covariance_data (np.array): covariance matrix of the data
            clipping_factor (float, optional): clipping factor. Defaults to 3.0.
            return_mask (bool, optional): whether to return the mask. Defaults to False.

        Returns:
            np.array: clipped data
        """
        inv_cov = np.linalg.inv(covariance_data)
        chi2 = []
        for i in range(len(xi_test)):
            res = xi_data - xi_test[i]
            chi2.append(np.dot(res, np.dot(inv_cov, res)))
        chi2 = np.asarray(chi2)
        c, low, upp = sigmaclip(chi2, low=clipping_factor, high=clipping_factor)
        mask = (chi2 > low) & (chi2 < upp)
        if return_mask:
            return xi_test[mask], mask
        return xi_test[mask]


def normalize_cov(cov):
    nbins = len(cov)
    corr = np.zeros_like(cov)
    for i in range(nbins):
        for j in range(nbins):
            corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    return corr
