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
        model_filters: Dict = {},
    ):
        self.theory_model = theory_model if isinstance(theory_model, Iterable) else [theory_model]
        self.model_filters = model_filters if isinstance(model_filters, Iterable) else [model_filters]
        self.fixed_parameters = fixed_parameters
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
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
        with torch.no_grad():
            prediction = []
            for model, filters in zip(self.theory_model, self.model_filters):
                pred = model.get_prediction(
                    x=torch.Tensor(theta),
                    filters=filters,
                )
                prediction.append(pred.numpy())
            prediction = np.concatenate(prediction, axis=-1)
            return prediction

    def log_likelihood(self, theta):
        """Log likelihood function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log likelihood
        """
        params = self.fill_params_batch(theta) if len(theta.shape) > 1 else self.fill_params(theta)
        prediction = self.get_model_prediction(params)
        diff = self.observation - prediction
        if len(theta.shape) > 1:
            return np.asarray([-0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(theta))])
        return -0.5 * diff @ self.precision_matrix @ diff.T

    def __call__(self, vectorize=True, random_state=0, precondition=True, **kwargs):
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

        self.sampler.run(n_total=10_000)

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