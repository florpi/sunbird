from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
import xarray as xr
from sunbird.models import Predictor
from sunbird.covariance import CovarianceMatrix, normalize_cov
from sunbird.abacus_utils.read_statistics import read_statistic, read_parameters
import sys

DATA_PATH = Path(__file__).parent.parent.parent / "data/different_hods/"

class Inference(ABC):
    def __init__(
        self,
        theory_model: Predictor,
        observation: np.array,
        covariance_matrix: np.array,
        priors: Dict,
        fixed_parameters: Dict[str, float],
        select_filters: Dict,
        slice_filters: Dict,
        output_dir: Path,
        device: str ="cpu",
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
        device: str ="cpu",
    )->"Inference":
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
        select_filters = config['select_filters']
        slice_filters = config['slice_filters']
        observation = cls.get_observation_for_abacus(
            cosmology= config['data']['cosmology'],
            hod_idx= config['data']['hod_idx'],
            statistics=config['data']['summaries'],
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        parameters = cls.get_parameters_for_abacus(
            cosmology= config['data']['cosmology'],
            hod_idx= config['data']['hod_idx'],
        )
        fixed_parameters = {}
        for k in config['fixed_parameters']:
            fixed_parameters[k] = parameters[k]
        covariance = CovarianceMatrix(
            statistics = config['data']['summaries'],
            select_filters=select_filters,
            slice_filters = slice_filters,
        )
        cov_data = covariance.get_covariance_data()
        cov_emulator_error = covariance.get_covariance_emulator_error()
        covariance_matrix = cov_data + cov_emulator_error
        theory_model = cls.get_theory_model(
            config["theory_model"], 
        )
        parameters_to_fit = [p for p in theory_model.parameters if p not in fixed_parameters.keys()]
        priors = cls.get_priors(config["priors"], parameters_to_fit)
        return cls(
            theory_model=theory_model,
            observation=observation,
            select_filters=select_filters,
            slice_filters=slice_filters,
            covariance_matrix=covariance_matrix,
            fixed_parameters=fixed_parameters,
            priors=priors,
            output_dir=config['inference']['output_dir'],
            device=device,
        )
    
    @classmethod
    def get_observation_for_abacus(
        cls,
        cosmology: int,
        hod_idx: int,
        statistics: str,
        select_filters: Dict,
        slice_filters: Dict,
    )->np.array:
        observation = []
        for statistic in statistics:
            observation.append(read_statistic(
                    statistic=statistic,
                    cosmology = cosmology,
                    dataset = 'different_hods',
                    select_filters=select_filters,
                    slice_filters = slice_filters,
                ).values[hod_idx].reshape(-1))
        return np.hstack(observation)

    @classmethod
    def get_parameters_for_abacus(
        cls,
        cosmology: int,
        hod_idx: int,
    ):
        return read_parameters(
            cosmology=cosmology,
            dataset='different_hods',
        ).iloc[hod_idx].to_dict()
 

    @classmethod
    def get_priors(cls, prior_config: Dict[str,Dict], parameters_to_fit: List[str])->Dict:
        distributions_module = importlib.import_module(prior_config.pop("stats_module"))
        prior_dict = {}
        for param in parameters_to_fit:
            config_for_param = prior_config[param]
            prior_dict[param] = cls.initialize_distribution(
                cls, distributions_module, config_for_param
            )
        return prior_dict

    @classmethod
    def get_observation(cls, path_to_observation: Path, filters: Dict)->np.array:
        return 

    @classmethod
    def get_covariance_data(cls, path_to_cov, s_min: float, quintiles):
        data = np.load(path_to_cov, allow_pickle=True).item()
        s = data["s"]
        multipoles = []
        if quintiles is not None:
            for q in quintiles:
                multipoles.append(data["multipoles"][:, q, 0][:, s > s_min])
        else:
            multipoles.append(data["multipoles"][:, 0][:, s > s_min])
        multipoles = np.array(multipoles)
        multipoles = np.transpose(multipoles, axes=(1, 0, 2))
        multipoles = multipoles.reshape((len(multipoles), -1))
        return np.cov(multipoles.T)

    @classmethod
    def get_theory_model(cls, theory_config,):
        module = theory_config.pop("module")
        class_name = theory_config.pop("class")
        module = getattr(importlib.import_module(module), class_name)
        if 'args' in theory_config:
            return module(**theory_config['args'],)
        return module()

    @abstractmethod
    def __call__(
        self,
    ):
        pass

    def invert_covariance(self, covariance_matrix,):
        return np.linalg.inv(covariance_matrix)

    def initialize_distribution(cls, distributions_module, dist_param):
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
        prediction,
    ):
        diff = prediction - self.observation
        return -0.5 * diff @ self.inverse_covariance_matrix @ diff

    def get_loglikelihood_for_prediction_vectorized(
        self,
        prediction,
    ):
        diff = prediction - self.observation
        right = np.einsum('ik,...k', self.inverse_covariance_matrix, diff)
        return -0.5 * np.einsum('ki,ji', diff, right)[:,0]

    def sample_from_prior(
        self, 
    ):
        params = {}
        for param, dist in self.priors.items():
            params[param] = dist.rvs()
        for p, v in self.fixed_parameters.items():
            params[p] = v
        return params, self.theory_model(params, select_filters=self.select_filters, slice_filters=self.slice_filters)

    def get_model_prediction(
        self,
        parameters,
    ):
        params = dict(zip(list(self.priors.keys()), parameters))
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param] 
        return self.theory_model(
            params,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )

    def get_model_prediction_vectorized(
        self,
        parameters,
    ):
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
