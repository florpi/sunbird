from typing import Dict, Optional
import importlib
import numpy as np
import jax.numpy as jnp
import emcee
import torch
import sys
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool


class MCMC:
    def __init__(
        self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        fixed_parameters: Dict[str, float] = {},
        model_filters: Dict = {},
    ):
        self.theory_model = theory_model
        self.fixed_parameters = fixed_parameters
        self.observation = observation
        self.priors = priors
        self.precision_matrix = precision_matrix
        self.model_filters = model_filters
        self.ndim = len(self.priors.keys()) - len(self.fixed_parameters.keys())

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

    def get_model_prediction(self, theta):
        """Get model prediction

        Args:
            theta (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        with torch.no_grad():
            pred = self.theory_model.get_prediction(
                x=torch.Tensor(theta),
                filters=self.model_filters,
            )
            return pred.numpy()

    def log_likelihood(self, theta):
        """Log likelihood function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log likelihood
        """
        params = self.fill_params(theta)
        prediction = self.get_model_prediction(params)
        diff = self.observation - prediction
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
        return 0.0

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
    
    def __call__(self, nwalkers=4, niter=1000, nthreads=1, burnin=200, check_every=200):
        """Run the MCMC sampler

        Args:
            nwalkers (int, optional): number of walkers. Defaults to 4.
            nsteps (int, optional): number of steps. Defaults to 1000.

        Returns:
            emcee.EnsembleSampler: emcee sampler object
        """
        ndim = len(self.priors.keys()) - len(self.fixed_parameters.keys())
        initial = self.init_params() + 1e-4 * np.random.randn(nwalkers, ndim)
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability
        )
        state = self.sampler.run_mcmc(initial, burnin, progress=True)
        self.sampler.reset()
        self.old_tau = np.inf
        for sample in self.sampler.sample(state, iterations=niter, progress=True):
            if self.sampler.iteration % check_every:
                continue
            if self.is_converged():
                print("Converged")
                break            
        return self.sampler

    def is_converged(self, stop_factor=100, relative_diff=0.01):
        tau = self.sampler.get_autocorr_time(tol=0)
        cond1 = np.all(tau * stop_factor < self.sampler.iteration)
        cond2 = np.all(np.abs(self.old_tau - tau) / tau < relative_diff)
        converged = cond1 and cond2
        self.old_tau = tau
        return converged

    def get_chain(self, **kwargs):
        """Get the chain from the sampler

        Returns:
            np.array: chain
        """
        return self.sampler.get_chain(**kwargs)
