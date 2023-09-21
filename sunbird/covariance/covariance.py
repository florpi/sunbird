import numpy as np
import json
from pathlib import Path
from scipy.stats import sigmaclip
from typing import List, Dict, Optional, Callable
from sunbird.data import data_readers

DATA_PATH = Path(__file__).parent.parent.parent / "data/"
MODEL_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"


class CovarianceMatrix:
    def __init__(
        self,
        statistics: List[str],
        slice_filters: Dict = None,
        select_filters: Dict = None,
        covariance_data_class: str = 'AbacusSmall',
        emulator_data_class: str = 'Abacus',
        dataset: str = 'bossprior',
        loss: str = 'learned_gaussian',
        output_transforms: Optional[Callable] = None,
        emulators=None,
        obs_config: Dict = {},
        path_to_models: Path = MODEL_PATH,
    ):
        """Compute a covariance matrix for a list of statistics and filters in any
        dimension

        Args:
            statistics (List[str]): list of statistics to use
            slice_filters (Dict): dictionary with slice filters on given coordinates
            select_filters (Dict): dictionary with select filters on given coordinates
        """
        self.dataset = dataset
        self.loss = loss
        self.data_reader = getattr(data_readers, covariance_data_class)(
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
            transforms=output_transforms,
            dataset=dataset,
            **obs_config.get("args", {}),
        )
        self.covariance_simulations_reader = getattr(data_readers, "AbacusSmall")(
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
            transforms=output_transforms,
            dataset=dataset,
        )
        self.training_simulations_reader = getattr(data_readers, emulator_data_class)(
            dataset=dataset,
            statistics=statistics,
            slice_filters=slice_filters,
            select_filters=select_filters,
        )
        self.covariance_data_class = covariance_data_class
        self.statistics = statistics
        self.slice_filters = slice_filters
        self.select_filters = select_filters
        self.emulators = emulators
        self.path_to_models = path_to_models

    def get_covariance_data(
        self,
        volume_scaling: float = None,
        fractional: bool = False,
        return_nmocks=False,
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
            volume_scaling = 1.0
        return self.estimate_covariance_from_data_reader(
            data_reader=self.data_reader,
            fractional=fractional,
            volume_scaling=volume_scaling,
            return_nmocks=return_nmocks,
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
                xi = self.training_simulations_reader.read_statistic(
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
                self.training_simulations_reader.get_all_parameters(
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
        if self.emulators is None:
            from sunbird.summaries import DensitySplitAuto, DensitySplitCross, TPCF, DensityPDF

            self.emulators = {
                "density_split_cross": DensitySplitCross(dataset=self.dataset, path_to_models=self.path_to_models, loss=self.loss),
                "density_split_auto": DensitySplitAuto(dataset=self.dataset, path_to_models=self.path_to_models, loss=self.loss),
                "tpcf": TPCF(dataset=self.dataset, path_to_models=self.path_to_models, loss=self.loss),
                "density_pdf": DensityPDF(dataset=self.dataset, path_to_models=self.path_to_models, loss=self.loss),
            }
        xi_model = []
        for statistic in self.statistics:
            pred, error = self.emulators[statistic].get_for_batch_inputs(
                inputs=inputs,
                select_filters=self.select_filters,
                slice_filters=self.slice_filters,
            )
            xi_model.append(pred)
        xi_model = np.hstack(xi_model)
        return np.squeeze(np.array(xi_model))

    def estimate_covariance_from_data_reader(
        self,
        data_reader: data_readers.DataReader,
        fractional: bool = False,
        volume_scaling: float = 1.0,
        return_nmocks: bool = False,
    ):
        """estimate covariance matrix from a set of simulations read by data_reader

        Args:
            data_reader (data_readers.DataReader): data reader, will load the necessary simulations
            apply_hartlap_correction (bool, optional): whether to apply hartlap correction.
            Defaults to True.
            fractional (bool, optional): whether to use fractional covariance.
            Defaults to False.
            volume_scaling (float, optional): volume scaling factor. Defaults to 1.0 (for a CMASS-like volume).

        Returns:
            np.array: covariance matrix
        """
        summaries = data_reader.gather_summaries_for_covariance()
        n_mocks = len(summaries)
        if fractional:
            cov = np.cov(summaries / np.mean(summaries, axis=0), rowvar=False)
        else:
            cov = np.cov(summaries, rowvar=False)
        if return_nmocks:
            return cov / volume_scaling, n_mocks
        return cov / volume_scaling

    def get_covariance_simulation(
        self,
        fractional: bool = False,
        return_nmocks: bool = False,
    ) -> np.array:
        """Get the covariance matrix associated with the finite volume
        of the simulations used to train the emulator.

        Args:
            apply_hartlap_correction (bool, optional): whether to apply hartlap correction. Defaults to True.
            fractional (bool, optional): whether to return the fractional covariance matrix. Defaults to False.

        Returns:
            np.array: covariance matrix of the simulations sample variance.
        """
        return self.estimate_covariance_from_data_reader(
            data_reader=self.covariance_simulations_reader,
            fractional=fractional,
            volume_scaling=64,
            return_nmocks=return_nmocks,
        )

    def get_covariance_emulator(
        self,
        fractional: bool = False,
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
        absolute_error = xi_model - xi_test
        if fractional:
            if return_mean:
                return np.cov(absolute_error / xi_test, rowvar=False), np.mean(
                    absolute_error / xi_test, axis=0
                )
            return np.cov(absolute_error / xi_test, rowvar=False)
        if return_mean:
            return np.cov(absolute_error, rowvar=False), np.mean(absolute_error, axis=0)
        return np.cov(absolute_error, rowvar=False)


def normalize_cov(cov):
    nbins = len(cov)
    corr = np.zeros_like(cov)
    for i in range(nbins):
        for j in range(nbins):
            corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    return corr
