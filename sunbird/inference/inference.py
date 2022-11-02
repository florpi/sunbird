from abc import ABC, abstractmethod
import pandas as pd
from scipy import stats
from pathlib import Path
import importlib
import numpy as np
import yaml


DATA_PATH = Path('/n/home11/ccuestalazaro/sunbird/data/')

class Inference(ABC):
    def __init__(
        self,
        theory_model,
        observation,
        covariance_matrix,
        priors,
        fixed_parameters,
        s_min: float,
        device = 'cpu',
    ):
        self.theory_model = theory_model
        self.observation = observation
        self.covariance_matrix = covariance_matrix
        self.priors = priors
        self.fixed_parameters = fixed_parameters
        self.device = device
        self.s_min = s_min
        self.param_names = list(self.priors.keys())

    @classmethod
    def from_config(cls, path_to_config, device="cpu", ):
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
        true_params_dict, observation = cls.get_observations(config["data"],)
        fixed_parameters = {k: true_params_dict[k] for k in config['data']['fixed_parameters']}
        priors = cls.get_priors(config["priors"], list(fixed_parameters.keys()))
        covariance_matrix = cls.get_covariance(s_min=config['data']['s_min'])
        theory_model = cls.get_theory_model(config["theory_model"],)
        return cls(
            theory_model=theory_model,
            observation=observation,
            covariance_matrix=covariance_matrix,
            fixed_parameters=fixed_parameters,
            priors=priors,
            device=device,
            s_min=config['data']['s_min'],
        )

    @classmethod
    def get_priors(cls, prior_config, fixed_parameters):
        distributions_module = importlib.import_module(prior_config.pop("stats_module"))
        parameters_to_fit = prior_config.keys()
        prior_dict = {}
        for param in parameters_to_fit:
            if param not in fixed_parameters:
                config_for_param = prior_config[param]
                prior_dict[param] = cls.initialize_distribution(
                    cls, distributions_module, config_for_param
                )
        return prior_dict

    def initialize_distribution(cls, distributions_module, dist_param):
        if dist_param["distribution"] == "uniform":
            max_uniform = dist_param.pop("max")
            min_uniform = dist_param.pop("min")
            dist_param["loc"] = min_uniform
            dist_param["scale"] = max_uniform - min_uniform
        dist = getattr(distributions_module, dist_param.pop("distribution"))
        return dist(**dist_param)


    @classmethod
    def get_observations(
        cls, data_config,
    ):
        cosmo_idx = data_config['cosmology']
        hod_idx = data_config['hod_idx']
        data = np.load(
            DATA_PATH / f'full_ap/clustering/ds_cross_xi_smu_zsplit_Rs20_c{str(cosmo_idx).zfill(3)}_ph000.npy', 
            allow_pickle=True,
        ).item()
        s = data['s']
        multipoles = []
        for quintile in [0,1,3,4]:
            multipoles += list(np.mean(data['multipoles'], axis=1)[hod_idx,quintile, 0][s > data_config['s_min']])
        multipoles = np.array(multipoles)
        true_params_dict = dict(pd.read_csv(DATA_PATH / f'full_ap/cosmologies/AbacusSummit_c{str(cosmo_idx).zfill(3)}_hod1000.csv').iloc[hod_idx])
        return true_params_dict, multipoles 

    @classmethod
    def get_covariance(cls, s_min: float):
        path_to_cov = DATA_PATH / 'full_ap/covariance/ds_cross_xi_smu_zsplit_Rs20_landyszalay_randomsX50.npy'
        data = np.load(path_to_cov, allow_pickle=True).item()
        s = data['s']
        multipoles = []
        for q in [0,1,3,4]:
            multipoles.append(data['multipoles'][:,q,0][:,s>s_min])
        multipoles = np.array(multipoles)
        multipoles = np.transpose(multipoles,axes=(1,0,2))
        multipoles = multipoles.reshape((len(multipoles),-1))
        return np.cov(multipoles.T)

    @classmethod
    def get_theory_model(cls, theory_config):
        module = theory_config.pop("module")
        class_name = theory_config.pop("class")
        return getattr(importlib.import_module(module), class_name)(
            **theory_config["params"]
        )

    @abstractmethod
    def __call__(self,):
        pass


    def get_loglikelihood_for_prediction(
        self, prediction, 
    ):
        return stats.multivariate_normal.logpdf(prediction, self.observation, self.covariance_matrix)

    def sample_from_prior(self,):
        params = {}
        for param, dist in self.priors.items():
            params[param] = dist.rvs()
        for p, v in self.fixed_parameters.items():
            params[p] = v
        return self.theory_model(params)

    def get_model_prediction(
        self, parameters,
    ):
        params = {}
        for i, param in enumerate(self.priors.keys()):
            params[param] = parameters[:,i]
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param] * np.ones(len(parameters))
        out = self.theory_model.get_for_batch(params, s_min=self.s_min)
        out = out.reshape((len(parameters),-1))
        return out
