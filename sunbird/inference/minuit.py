from iminuit import Minuit
import numpy as np
import torch
from typing import Dict, Optional
from collections.abc import Iterable
import logging


class MinuitProfiler:
    def __init__(
        self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        start: Dict[str, float],
        ranges: Optional[Dict[str, tuple]] = {},
        labels: Dict[str, str] = {},
        fixed_params: Dict[str, float] = {},
        model_filters: Dict = {},
    ):
        self.theory_model = theory_model if isinstance(theory_model, Iterable) else [theory_model]
        self.model_filters = model_filters if isinstance(model_filters, Iterable) else [model_filters]
        self.fixed_params = fixed_params
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
        self.logger = logging.getLogger('MinuitProfiler')
        self.logger.info('Initializing MinuitProfiler.')
        self.logger.info(f'Free parameters: {[key for key in priors.keys() if key not in fixed_params.keys()]}')
        self.logger.info(f'Fixed parameters: {[key for key in priors.keys() if key in fixed_params.keys()]}')

        minuit_params = {}
        minuit_params['name'] = [str(param) for param in start.keys()]

        self.minuit = Minuit(self.log_likelihood_minuit, **start, **minuit_params)

        for param in start.keys():
            self.minuit.limits[param] = (ranges[param][0], ranges[param][1])

        self.minuit.errordef = Minuit.LIKELIHOOD
        

    def minimize(self):
        self.minuit.migrad(ncall=int(1e5))
        self.minuit.hesse()
        print(self.minuit.params)
        print(self.minuit.valid)

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
            if param not in self.fixed_params.keys():
                params[i] = theta[itheta]
                itheta += 1
            else:
                params[i] = self.fixed_params[param]
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
        theta = np.array(theta)
        params = self.fill_params_batch(theta) if len(theta.shape) > 1 else self.fill_params(theta)
        prediction = self.get_model_prediction(params)
        diff = self.observation - prediction
        if len(theta.shape) > 1:
            return [0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(theta))]
        return 0.5 * diff @ self.precision_matrix @ diff.T

    def log_likelihood_minuit(self, *theta):
        return self.log_likelihood(theta)

    def log_prior(self, theta):
        """Log prior function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log prior
        """
        lp = np.zeros(len(theta))
        itheta = 0
        for i, param in enumerate(self.priors.keys()):
            if param not in self.fixed_params.keys():
                dist = self.priors[param]
                lp[itheta] = dist.logpdf(theta[itheta])
                itheta += 1
            else:
                continue
        if any(~np.isfinite(lp)):
            return -np.inf
        return np.sum(lp)

    def log_posterior(self, theta):
        """Log (posterior) probability function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log posterior probability
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)