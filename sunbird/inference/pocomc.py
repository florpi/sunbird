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
    ):
        self.theory_model = theory_model if isinstance(theory_model, list) else [theory_model]
        self.select_filters = select_filters if isinstance(select_filters, list) else [select_filters]
        self.slice_filters = slice_filters if isinstance(slice_filters, list) else [slice_filters]
        self.coordinates = coordinates if isinstance(coordinates, list) else [coordinates]
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

    def apply_model_filters(self, prediction, coordinates,
        select_filters, slice_filters, batch=False):
        """Apply filters to the model prediction.
        
        Args:
            prediction (np.array): model prediction
            coordinates (dict): coordinates of the model prediction
            select_filters (dict): select filters
            slice_filters (dict): slice filters
            
            Returns:
                np.array: filtered prediction
        """
        coords = coordinates.copy()
        if batch:
            coords_shape = tuple(len(v) for k, v in coords.copy().items())
            dimensions = ["batch"] + list(coords.keys())
            coords["batch"] = range(len(prediction))
            prediction = prediction.reshape((len(prediction), *coords_shape))
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=select_filters, slice_filters=slice_filters
            ).values.reshape(len(prediction), -1)
        else:
            coords_shape = tuple(len(v) for k, v in coords.items())
            prediction = prediction.reshape(coords_shape)
            dimensions = list(coords.keys())
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=select_filters, slice_filters=slice_filters
            ).values.reshape(-1)

    def get_model_prediction(self, theta, batch=False):
        """Get model prediction

        Args:
            theta (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        with torch.no_grad():
            prediction = []
            for i, model in enumerate(self.theory_model):
                pred = model.get_prediction(
                    x=torch.Tensor(theta),
                )
                if self.select_filters or self.slice_filters:
                    pred = self.apply_model_filters(
                        prediction=pred,
                        coordinates=self.coordinates[i],
                        select_filters=self.select_filters[i],
                        slice_filters=self.slice_filters[i],
                        batch=batch
                    )
                prediction.append(pred)
            prediction = np.concatenate(prediction, axis=-1)
            return prediction

    def log_likelihood(self, theta):
        """Log likelihood function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log likelihood
        """
        batch = len(theta.shape) > 1
        params = self.fill_params_batch(theta) if batch else self.fill_params(theta)
        prediction = self.get_model_prediction(params, batch=batch)
        diff = self.observation - prediction
        # print(np.min(prediction), np.max(prediction))
        if len(theta.shape) > 1:
            return np.asarray([-0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(theta))])
        return -0.5 * diff @ self.precision_matrix @ diff.T

    def __call__(self, vectorize=True, random_state=0, precondition=True, n_total=4096, **kwargs):
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

        self.sampler.run(progress=True, n_total=n_total)

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