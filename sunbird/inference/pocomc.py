import pocomc
from typing import Dict, Optional
from collections.abc import Iterable
import importlib
import numpy as np
import jax.numpy as jnp
import torch
import sys
import time
import logging
from sunbird.inference.base import BaseSampler
from sunbird.data.data_utils import convert_to_summary
from sunbird.inference.priors import AbacusSummitEllipsoid


class PocoMCSampler(BaseSampler):
    def __init__(
        self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        ranges: Optional[Dict[str, tuple]] = {},
        labels: Dict[str, str] = {},
        fixed_parameters: Dict[str, float] = {},
        slice_filters: Dict = {},
        select_filters: Dict = {},
        coordinates: list = [],
        ellipsoid: bool = False,
    ):
        self.theory_model = theory_model
        self.fixed_parameters = fixed_parameters
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
        self.ellipsoid = ellipsoid
        if self.ellipsoid:
            self.abacus_ellipsoid = AbacusSummitEllipsoid()
        self.ndim = len(self.priors.keys()) - len(self.fixed_parameters.keys())
        self.logger = logging.getLogger('PocoMCSampler')
        self.logger.info('Initializing PocoMCSampler.')
        self.logger.info(f'Free parameters: {[key for key in priors.keys() if key not in fixed_parameters.keys()]}')
        self.logger.info(f'Fixed parameters: {[key for key in priors.keys() if key in fixed_parameters.keys()]}')
        super().__init__()

    def fill_params(self, theta):
        """Fill the parameter vector to include fixed parameters

        Args:
            theta (np.array): input parameters

        Returns:
            np.array: filled parameters
        """
        params = np.ones(len(self.priors.keys()))
        itheta = 0
        for i, param in enumerate(self.priors.keys()):
            if param not in self.fixed_parameters.keys():
                params[i] = theta[itheta]
                itheta += 1
            else:
                params[i] = self.fixed_parameters[param]
        return params

    def fill_params_batch(self, thetas):
        """Fill the batch of parameter vectors to include fixed parameters

        Args:
            thetas (np.array): input parameters

        Returns:
            np.array: filled parameters
        """
        params = np.ones((len(thetas), len(self.priors.keys())))
        for i, theta in enumerate(thetas):
            params[i] = self.fill_params(theta)
        return params

    def get_model_prediction(self, theta):
        """Get model prediction

        Args:
            theta (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        return self.theory_model.get_prediction(x=theta)

    def log_likelihood(self, theta):
        """Log likelihood function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log likelihood
        """
        batch = len(theta.shape) > 1
        params = self.fill_params_batch(theta) if batch else self.fill_params(theta)
        prediction = self.get_model_prediction(params)
        diff = self.observation - prediction
        if batch:
            logl = np.asarray([-0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(theta))])
            if self.ellipsoid:
                logl += np.asarray([self.abacus_ellipsoid.log_likelihood(params[i, :8]) for i in range(len(theta))])
        else:
            logl = -0.5 * diff @ self.precision_matrix @ diff.T
            if self.ellipsoid:
                logl += self.abacus_ellipsoid.log_likelihood(params[:8])
        return logl

    def __call__(self, vectorize=True, random_state=0, precondition=True, n_total=4096, progress=True, **kwargs):
        """Run the sampler

        Args:
            vectorize (bool, optional): Vectorize the log likelihood call. Defaults to False.
            random_state (int, optional): Random seed. Defaults to 0.
            precondition (bool, optional): If False, use standard MCMC without normalizing flow. Defaults to True.
            kwargs: Additional arguments for the sampler
        """
        prior = pocomc.Prior([value for key, value in self.priors.items() if key not in self.fixed_parameters.keys()])

        self.sampler = pocomc.Sampler(
            likelihood=self.log_likelihood,
            prior=prior,
            vectorize=vectorize,
            random_state=random_state,
            precondition=precondition,
            **kwargs,
        )

        self.sampler.run(progress=progress, n_total=n_total)

    def get_chain(self, **kwargs):
        """Get the chain from the sampler

        Returns:
            np.array: chain
        """
        samples, weights, logl, logp = self.sampler.posterior()
        return {'samples': samples, 'weights': weights,
                'log_likelihood': logl, 'log_prior': logp}


if __name__ == "__main__":
    PocoMCSampler()