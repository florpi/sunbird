from typing import Dict, Optional
from collections.abc import Iterable
import importlib
import numpy as np
import jax.numpy as jnp
import emcee
import torch
import sys
import time
import logging
from sunbird.inference.base import BaseSampler


class EmceeSampler(BaseSampler):
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
        self.logger = logging.getLogger('EmceeSampler')
        self.logger.info('Initializing EmceeSampler.')
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
            return [-0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(theta))]
        return -0.5 * diff @ self.precision_matrix @ diff.T

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
            if param not in self.fixed_parameters.keys():
                dist = self.priors[param]
                lp[itheta] = dist.logpdf(theta[itheta])
                itheta += 1
            else:
                continue
        if any(~np.isfinite(lp)):
            return -np.inf
        return np.sum(lp)

    def log_probability(self, theta):
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

    def log_probability_batch(self, thetas):
        """Batch log probability function

        Args:
            thetas (np.array): input parameters

        Returns:
            np.array: log posterior probability
        """
        log_prior = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            log_prior[i] = self.log_prior(theta)
        log_like = self.log_likelihood(thetas)
        return log_prior + log_like

    def init_params(self):
        """Initialize parameters

        Returns:
            np.array: initial parameters
        """
        params = np.ones(self.ndim)
        itheta = 0
        for i, param in enumerate(self.priors.keys()):
            if param not in self.fixed_parameters.keys():
                dist = self.priors[param]
                params[itheta] = dist.rvs()
                itheta += 1
            else:
                continue
        return params
    
    def __call__(self, nwalkers=4, niter=1000, nthreads=1, burnin=500, check_every=100,
        vectorize=False, moves=None):
        """Run the MCMC sampler

        Args:
            nwalkers (int, optional): number of walkers. Defaults to 4.
            nsteps (int, optional): number of steps. Defaults to 1000.

        Returns:
            emcee.EnsembleSampler: emcee sampler object
        """
        ndim = len(self.priors.keys()) - len(self.fixed_parameters.keys())
        initial = self.init_params() + 1e-4 * np.random.randn(nwalkers, ndim)
        log_prob_fn = self.log_probability_batch if vectorize else self.log_probability 
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_fn,
            vectorize=vectorize, moves=moves,
        )
        self.logger.info('Running burn-in.')
        state = self.sampler.run_mcmc(initial, burnin, progress=False)
        self.sampler.reset()
        self.old_tau = np.inf
        for sample in self.sampler.sample(state, iterations=niter, progress=False):
            if self.sampler.iteration % check_every:
                continue
            converged, tau = self.check_convergence()
            accept = 100 * np.mean(self.sampler.acceptance_fraction)
            least_converged = 100 * np.max(np.abs(self.old_tau - tau) / tau)
            self.logger.info(f"Iteration {self.sampler.iteration} / Acceptance rate {accept:.2f}% / Max. autocorr. {np.max(tau):.2f} / Least converged {least_converged:.2f}%")
            if converged:
                self.logger.info("Converged")
                break            
            self.old_tau = tau
        return self.sampler

    def check_convergence(self, stop_factor=10, relative_diff=0.01):
        tau = self.sampler.get_autocorr_time(tol=0)
        cond1 = np.all(tau * stop_factor < self.sampler.iteration)
        cond2 = np.all(np.abs(self.old_tau - tau) / tau < relative_diff)
        converged = cond1 & cond2
        return converged, tau

    def get_chain(self, **kwargs):
        """Get the chain from the sampler

        Returns:
            np.array: chain
        """
        samples = self.sampler.get_chain(**kwargs)
        weights = logl = logp = None
        samples, weights, logl, logp = self.sampler.posterior()
        return {'samples': samples, 'weights': weights,
                'log_likelihood': logl, 'log_posterior': logp}
