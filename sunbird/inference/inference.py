from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import numpy as np
import yaml
from typing import Dict, List, Tuple
from sunbird.covariance import CovarianceMatrix
from sunbird.read_utils.read_statistics import (
    read_statistic_abacus, read_parameters_abacus,
    read_statistic_patchy, read_parameters_patchy
)
import sys
import matplotlib.pyplot as plt


class Inference(ABC):
    def __init__(
        self,
        theory_model: "Summary",
        observation: np.array,
        covariance_matrix: np.array,
        priors: Dict,
        fixed_parameters: Dict[str, float],
        select_filters: Dict,
        slice_filters: Dict,
        output_dir: Path,
        device: str = "cpu",
    ):
        self.theory_model = theory_model
        self.observation = observation
        self.covariance_matrix = covariance_matrix
        self.inverse_covariance_matrix = self.invert_covariance(
            covariance_matrix=self.covariance_matrix,
        )
        self.priors = priors
        self.n_dim = len(self.priors)
        self.fixed_parameters = fixed_parameters
        self.device = device
        self.param_names = list(self.priors.keys())
        self.select_filters = select_filters
        self.slice_filters = slice_filters
        self.output_dir = Path(output_dir)

    @classmethod
    def from_abacus_config(
        cls,
        path_to_config: Path,
        device: str = "cpu",
    ) -> "Inference":
        """Read from config file to fit one of the abacus summit
        simulations

        Args:
            path_to_config (Path): path to configuration file
            device (str, optional): device to use to run model. Defaults to "cpu".

        Returns:
            Inference: inference object
        """
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_abacus_config_dict(
            config=config, device=device,
        )

    @classmethod
    def from_patchy_config(
        cls,
        path_to_config: Path,
        device: str = "cpu",
    ) -> "Inference":
        """Read from config file to fit one of the BOSS Patchy
        mocks

        Args:
            path_to_config (Path): path to configuration file
            device (str, optional): device to use to run model. Defaults to "cpu".

        Returns:
            Inference: inference object
        """
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_patchy_config_dict(
            config=config, device=device,
        )

    @classmethod
    def from_abacus_config_dict(cls, config: Dict, device: str = "cpu"):
        """Use dictionary config to fit one of the abacus summit
        simulations

        Args:
            config (Dict): dictionary with configuration
            device (str, optional): device to use to run model. Defaults to "cpu".

        Returns:
            Inference: inference object
        """
        select_filters = config["select_filters"]
        slice_filters = config["slice_filters"]
        observation = cls.get_observation_for_abacus(
            cosmology=config["data"]["cosmology"],
            hod_idx=config["data"]["hod_idx"],
            statistics=config["data"]["summaries"],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        parameters = cls.get_parameters_for_abacus(
            cosmology=config["data"]["cosmology"],
            hod_idx=config["data"]["hod_idx"],
        )
        fixed_parameters = {}
        for k in config["fixed_parameters"]:
            fixed_parameters[k] = parameters[k]
        covariance_matrix = cls.get_covariance_matrix(
            statistics=config["data"]["summaries"],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        theory_model = cls.get_theory_model(
            config["theory_model"],
        )
        parameters_to_fit = [
            p for p in theory_model.parameters if p not in fixed_parameters.keys()
        ]
        priors = cls.get_priors(config["priors"], parameters_to_fit)
        return cls(
            theory_model=theory_model,
            observation=observation,
            select_filters=select_filters,
            slice_filters=slice_filters,
            covariance_matrix=covariance_matrix,
            fixed_parameters=fixed_parameters,
            priors=priors,
            output_dir=config["inference"]["output_dir"],
            device=device,
        )

    @classmethod
    def from_patchy_config_dict(cls, config: Dict, device: str = "cpu"):
        """Use dictionary config to fit one of the abacus summit
        simulations

        Args:
            config (Dict): dictionary with configuration
            device (str, optional): device to use to run model. Defaults to "cpu".

        Returns:
            Inference: inference object
        """
        select_filters = config["select_filters"]
        slice_filters = config["slice_filters"]
        observation = cls.get_observation_for_patchy(
            phase=config["data"]["phase"],
            statistics=config["data"]["summaries"],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        parameters = cls.get_parameters_for_patchy(
        )
        fixed_parameters = {}
        for k in config["fixed_parameters"]:
            fixed_parameters[k] = parameters[k]
        covariance_matrix = cls.get_covariance_matrix(
            statistics=config["data"]["summaries"],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        theory_model = cls.get_theory_model(
            config["theory_model"],
        )
        parameters_to_fit = [
            p for p in theory_model.parameters if p not in fixed_parameters.keys()
        ]
        priors = cls.get_priors(config["priors"], parameters_to_fit)
        return cls(
            theory_model=theory_model,
            observation=observation,
            select_filters=select_filters,
            slice_filters=slice_filters,
            covariance_matrix=covariance_matrix,
            fixed_parameters=fixed_parameters,
            priors=priors,
            output_dir=config["inference"]["output_dir"],
            device=device,
        )

    @classmethod
    def get_observation_for_abacus(
        cls,
        cosmology: int,
        hod_idx: int,
        statistics: List[str],
        select_filters: Dict,
        slice_filters: Dict,
    ) -> np.array:
        """Use one of the cosmology boxes from abacus summit
        latin hypercube as a mock observation. Select the hod sample
        ```hod_idx''' for a given statistic

        Args:
            cosmology (int): cosmology box to use as mock observation
            hod_idx (int): id of the hod sample for the given cosmology
            statistics (str): list of statistics to use (the statistic has
            to be one of either tpcf, density_split_auto or density_split_cross)
            select_filters (Dict): dictionary with filters to select values
            across a particular dimension
            slice_filters (Dict): dictionary with filters to slice values across
            a particular dimension

        Returns:
            np.array: array with observations
        """
        observation = []
        for statistic in statistics:
            observation.append(
                read_statistic_abacus(
                    statistic=statistic,
                    cosmology=cosmology,
                    dataset="different_hods_linsigma",
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                )
                .values[hod_idx]
                .reshape(-1)
            )
        return np.hstack(observation)

    @classmethod
    def get_observation_for_patchy(
        cls,
        phase: int,
        statistics: List[str],
        select_filters: Dict,
        slice_filters: Dict,
    ) -> np.array:
        """Use one of the phases (realizations) from the BOSS Patchy
        simulations as a mock observation.

        Args:
            phase (int): id of the Patchy mock realization
            statistics (str): list of statistics to use (the statistic has
            to be one of either tpcf, density_split_auto or density_split_cross)
            select_filters (Dict): dictionary with filters to select values
            across a particular dimension
            slice_filters (Dict): dictionary with filters to slice values across
            a particular dimension

        Returns:
            np.array: array with observations
        """
        observation = []
        for statistic in statistics:
            observation.append(
                (read_statistic_patchy(
                    statistic=statistic,
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                )
                .values)[phase]#.mean(axis=0)
                .reshape(-1)
            )
        return np.hstack(observation)

    @classmethod
    def get_observation_for_patchy_mean(
        cls,
        statistics: List[str],
        select_filters: Dict,
        slice_filters: Dict,
    ) -> np.array:
        """Use the mean of the MD-Patchy mocks (averaged across all
        realizations) as a mock observation.

        Args:
            statistics (str): list of statistics to use (the statistic has
            to be one of either tpcf, density_split_auto or density_split_cross)
            select_filters (Dict): dictionary with filters to select values
            across a particular dimension
            slice_filters (Dict): dictionary with filters to slice values across
            a particular dimension

        Returns:
            np.array: array with observations
        """
        observation = []
        for statistic in statistics:
            observation.append(
                (read_statistic_patchy(
                    statistic=statistic,
                    select_filters=select_filters,
                    slice_filters=slice_filters,
                )
                .values).mean(axis=0)
                .reshape(-1)
            )
        return np.hstack(observation)


    @classmethod
    def get_parameters_for_abacus(
        cls,
        cosmology: int,
        hod_idx: int,
    ) -> Dict[str, float]:
        """Read the parameters of an abacus summit simmulation

        Args:
            cosmology (int): cosmology model to read
            hod_idx (int): idx of the hod

        Returns:
            Dict: dictionary of parameters describing a simulation
        """
        return (
            read_parameters_abacus(
                cosmology=cosmology,
                dataset="different_hods_linsigma",
            )
            .iloc[hod_idx]
            .to_dict()
        )

    @classmethod
    def get_parameters_for_patchy(
        cls,
    ) -> Dict[str, float]:
        """Read the parameters of the BOSS Patchy mocks

        Returns:
            Dict: dictionary of parameters describing a simulation
        """
        return (
            read_parameters_patchy(
            )
            .iloc[0]
            .to_dict()
        )

    @classmethod
    def get_covariance_matrix(
        cls,
        statistics: List[str],
        select_filters: Dict,
        slice_filters: Dict,
        add_emulator_error: bool = True,
        apply_hartlap_correction=True,
    ) -> np.array:
        """Compute covariance matrix for a list of statistics

        Args:
            statistics (List[str]): list of statistics
            select_filters (Dict): filters to select values along a dimension
            slice_filters (Dict): filters to slice values along a dimension
            add_emulator_error (bool, optional): whether to add in the estimated emulator error. Defaults to True.
            apply_hartlap_correction (bool, optional): whether to correct the covariance matrix with the Hartlap factor
            to ensure the inverse covariance matrix is an unbiased estimator.

        Returns:
            np.array: covariance matrix
        """
        covariance = CovarianceMatrix(
            statistics=statistics,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        covariance_data = covariance.get_covariance_data(
            apply_hartlap_correction=apply_hartlap_correction
        )
        covariance_data = covariance_data
        if add_emulator_error:
            cov_emulator_error = covariance.get_covariance_emulator_error()
            return covariance_data + cov_emulator_error
        return covariance_data

    @classmethod
    def get_priors(
        cls, prior_config: Dict[str, Dict], parameters_to_fit: List[str]
    ) -> Dict:
        """Initialize priors for a given configuration and a list of parameters to fit

        Args:
            prior_config (Dict[str, Dict]): configuration of priors
            parameters_to_fit (List[str]): list of parameteters that are being fitted

        Returns:
            Dict: dictionary with initialized priors
        """
        distributions_module = importlib.import_module(prior_config.pop("stats_module"))
        prior_dict = {}
        for param in parameters_to_fit:
            config_for_param = prior_config[param]
            prior_dict[param] = cls.initialize_distribution(
                cls, distributions_module, config_for_param
            )
        return prior_dict

    @classmethod
    def get_theory_model(
        cls,
        theory_config: Dict,
    ) -> "Summary":
        """Get theory model

        Args:
            theory_config (Dict): configuration for theory model, both module and class

        Returns:
            Summary: summary to fit
        """
        module = theory_config.pop("module")
        class_name = theory_config.pop("class")
        module = getattr(importlib.import_module(module), class_name)
        if "args" in theory_config:
            return module(
                **theory_config["args"],
            )
        return module()

    @abstractmethod
    def __call__(
        self,
    ):
        pass

    def invert_covariance(
        self,
        covariance_matrix: np.array,
    ) -> np.array:
        """invert covariance matrix

        Args:
            covariance_matrix (np.array): covariance matrix to invert

        Returns:
            np.array: inverse covariance
        """
        return np.linalg.inv(covariance_matrix)

    def initialize_distribution(
        cls, distributions_module, dist_param: Dict[str, float]
    ):
        """Initialize a given prior distribution fromt he distributions_module

        Args:
            distributions_module : module form which to import distributions
            dist_param (Dict[str, float]): parameters of the distributions

        Returns:
            prior distirbution
        """
        if dist_param["distribution"] == "uniform":
            max_uniform = dist_param.pop("max")
            min_uniform = dist_param.pop("min")
            dist_param["loc"] = min_uniform
            dist_param["scale"] = max_uniform - min_uniform
        if dist_param["distribution"] == "norm":
            mean_gaussian = dist_param.pop("mean")
            dispersion_gaussian = dist_param.pop("dispersion")
            dist_param["loc"] = mean_gaussian
            dist_param["scale"] = dispersion_gaussian
        dist = getattr(distributions_module, dist_param.pop("distribution"))
        return dist(**dist_param)

    def get_loglikelihood_for_prediction(
        self,
        prediction: np.array,
    ) -> float:
        """Get gaussian loglikelihood for prediction

        Args:
            prediction (np.array): model prediction

        Returns:
            float: log likelihood
        """
        diff = prediction - self.observation
        return -0.5 * diff @ self.inverse_covariance_matrix @ diff

    def get_loglikelihood_for_prediction_vectorized(
        self,
        prediction: np.array,
    ) -> np.array:
        """Get vectorized loglikelihood prediction

        Args:
            prediction (np.array): prediciton in batches

        Returns:
            np.array: array of likelihoods
        """
        diff = prediction - self.observation
        right = np.einsum("ik,...k", self.inverse_covariance_matrix, diff)
        return -0.5 * np.einsum("ki,ji", diff, right)[:, 0]

    def sample_from_prior(
        self,
    ) -> Tuple:
        """Sample predictions from prior

        Returns:
            Tuple: tuple of parameters and theory model predictions
        """
        params = {}
        for param, dist in self.priors.items():
            params[param] = dist.rvs()
        for p, v in self.fixed_parameters.items():
            params[p] = v
        return params, self.theory_model(
            params, select_filters=self.select_filters, slice_filters=self.slice_filters
        )

    def get_model_prediction(
        self,
        parameters: np.array,
    ) -> np.array:
        """Get model prediction for a given set of input parameters

        Args:
            parameters (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        params = dict(zip(list(self.priors.keys()), parameters))
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param]
        model = self.theory_model(
            params,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )
        return self.theory_model(
            params,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )

    def get_model_prediction_vectorized(
        self,
        parameters: np.array,
    ) -> np.array:
        """get vectorized model predictions

        Args:
            parameters (np.array): input parameters

        Returns:
            np.array: model predictions in batches
        """
        params = {}
        for i, param in enumerate(self.priors.keys()):
            params[param] = parameters[:, i]
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param] * np.ones(
                len(parameters)
            )
        out = self.theory_model.get_for_batch(
            params,
            s_min=self.s_min,
        )
        return out.reshape((len(parameters), -1))
