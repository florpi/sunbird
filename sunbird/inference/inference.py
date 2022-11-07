from abc import ABC, abstractmethod
import pandas as pd
from scipy import stats
from pathlib import Path
import importlib
import numpy as np
import yaml
import json

# TODO:
# - Run mcmc with only HOD parameters
# - Run mcmc s > 0, s > 25

DATA_PATH = Path(__file__).parent.parent.parent / "data/"

class Inference(ABC):
    def __init__(
        self,
        theory_model,
        observation,
        covariance_matrix,
        priors,
        fixed_parameters,
        s_min: float,
        quintiles,
        output_dir,
        device="cpu",
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
        self.s_min = s_min
        self.param_names = list(self.priors.keys())
        self.quintiles = quintiles
        self.output_dir = Path(output_dir)

    @classmethod
    def from_config(
        cls,
        path_to_config,
        device="cpu",
    ):
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
        data_config = config["data"]
        observation = cls.get_observation(
            path_to_observation=data_config["path_to_observation"],
            quintiles=data_config["quintiles"],
            s_min=data_config["s_min"],
        )
        fixed_parameters = cls.get_fixed_parameters(
            path_to_fixed_parameters=data_config["path_to_fixed_parameters"]
        )
        fixed_parameters = {
            k: fixed_parameters[k] for k in data_config["fixed_parameters"]
        }
        covariance_matrix = cls.get_covariance(
            path_to_cov_data=data_config["path_to_covariance_data"],
            quintiles=data_config["quintiles"],
            s_min=data_config["s_min"],
        )
        theory_model = cls.get_theory_model(
            config["theory_model"],
        )
        parameters_to_fit = [p for p in theory_model.parameters if p not in fixed_parameters.keys()]
        priors = cls.get_priors(config["priors"], parameters_to_fit)
        return cls(
            theory_model=theory_model,
            observation=observation,
            covariance_matrix=covariance_matrix,
            fixed_parameters=fixed_parameters,
            priors=priors,
            device=device,
            s_min=data_config["s_min"],
            quintiles=data_config["quintiles"],
            output_dir = config['inference']['output_dir'],
        )

    @classmethod
    def get_priors(cls, prior_config, parameters_to_fit):
        distributions_module = importlib.import_module(prior_config.pop("stats_module"))
        prior_dict = {}
        for param in parameters_to_fit:
            config_for_param = prior_config[param]
            prior_dict[param] = cls.initialize_distribution(
                cls, distributions_module, config_for_param
            )
        return prior_dict

    @classmethod
    def get_fixed_parameters(cls, path_to_fixed_parameters):
        with open(path_to_fixed_parameters, "r") as fp:
            fixed_params = json.load(fp)
        return fixed_params

    @classmethod
    def get_observation(cls, path_to_observation, quintiles, s_min):
        data = np.load(
            path_to_observation,
        )
        s = data[0]
        multipoles = data[1:, :]
        multipoles = multipoles[quintiles]
        return multipoles[:, s > s_min].reshape(-1)

    @classmethod
    def get_covariance(cls, path_to_cov_data, s_min: float, quintiles):
        data = np.load(path_to_cov_data, allow_pickle=True).item()
        s = data["s"]
        multipoles = []
        for q in quintiles:
            multipoles.append(data["multipoles"][:, q, 0][:, s > s_min])
        multipoles = np.array(multipoles)
        multipoles = np.transpose(multipoles, axes=(1, 0, 2))
        multipoles = multipoles.reshape((len(multipoles), -1))
        return np.cov(multipoles.T)

    @classmethod
    def get_theory_model(cls, theory_config):
        module = theory_config.pop("module")
        class_name = theory_config.pop("class")
        return getattr(importlib.import_module(module), class_name)(
            **theory_config["params"]
        )

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
        return params, self.theory_model(params, quintiles=self.quintiles, s_min=self.s_min)

    def get_model_prediction(
        self,
        parameters,
    ):
        params = dict(zip(list(self.priors.keys()), parameters))
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param] 
        return self.theory_model(
            params,
            s_min=self.s_min,
            quintiles=self.quintiles,
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
            quintiles=self.quintiles,
        )
        return out.reshape((len(parameters), -1))
