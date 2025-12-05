import time
import logging
import torch
import numpy as np
from iminuit import Minuit
from typing import Dict, Optional
from collections.abc import Iterable


class MinuitProfiler:
    def __init__(
        self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        ranges: Optional[Dict[str, tuple]] = {},
        labels: Dict[str, str] = {},
        fixed_params: Dict[str, float] = {},
        model_filters: Dict = {},
    ):
        self.theory_model = theory_model if isinstance(theory_model, Iterable) else [theory_model]
        self.model_filters = model_filters if isinstance(model_filters, Iterable) else [model_filters]
        self.fixed_params = fixed_params
        self.varied_params = [param for param in priors.keys() if param not in fixed_params.keys()]
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
        self.logger = logging.getLogger('MinuitProfiler')
        self.logger.info('Initializing MinuitProfiler.')
        self.logger.info(f'Free parameters: {self.varied_params}')
        self.logger.info(f'Fixed parameters: {[key for key in priors.keys() if key in fixed_params.keys()]}')

        self.minuit_params = {}
        self.minuit_params['name'] = self.varied_params

    def get_start(self, limits=None):
        if limits is None:
            limits = self.ranges
        start = {key: np.random.uniform(limits[key][0], limits[key][1])
                 for key in self.varied_params}
        return start

    def _minimize_one(self, start):
        t0 = time.time()
        profile = {}
        minuit = Minuit(self.log_likelihood_minuit, **start, **self.minuit_params)
        for param in start.keys():
            minuit.limits[param] = (self.ranges[param][0], self.ranges[param][1])
        minuit.errordef = Minuit.LIKELIHOOD
        minuit.migrad(ncall=int(1e5))
        minuit.hesse()
        bestfit = {param: minuit.values[param] for param in self.varied_params}
        errors = {param: minuit.errors[param] for param in self.varied_params}
        chi2 = self.log_likelihood(list(bestfit.values()))
        profile['bestfit'] = bestfit
        profile['errors'] = errors
        profile['chi2'] = chi2
        # self.logger.info(f'Minimization took {time.time() - t0:.2f} s.')
        return profile
        
    def minimize(self, niter=1, nstart=1, sigma_iter=3):
        # profiles = []
        # for j in range(nstart):
        #     profiles.append(self._minimize_one(start=self.get_start()))
        profiles_glob = []
        for i in range(niter):
            profiles = []
            for j in range(nstart):
                limits = self.ranges if i == 0 else limits
                profiles.append(self._minimize_one(self.get_start(limits=limits)))
            profiles = sorted(profiles, key=lambda x: x['chi2'])[0]
            limits = {key: (profiles['bestfit'][key] - sigma_iter * profiles['errors'][key],
                            profiles['bestfit'][key] + sigma_iter * profiles['errors'][key])
                      for key in self.varied_params}
            profiles_glob.append(profiles)
            self.logger.info(f'Iteration {i+1}/{niter}: best chi2 = {profiles["chi2"]}')
        return profiles_glob

        # profiles = [self._minimize_one(start=self.get_start()) for i in range(nstart)]
        # profiles = sorted(profiles, key=lambda x: x['chi2'])[0]
        # # self.logger.info(f'Iteration {i+1}/{niter}: best chi2 = {profiles["chi2"]}')
        # start = profiles['bestfit']
        # profiles = self._minimize_one(start=start)
        # return profiles

        # if nstart == 1:
        #     return self._minimize_one(start=self.get_start())
        # else:
        #     profiles = []
        #     for i in range(nstart):
        #         profiles.append(self._minimize_one(start=self.get_start()))
        #     profiles = sorted(profiles, key=lambda x: x['chi2'])
        #     return profiles

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