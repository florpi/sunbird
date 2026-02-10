"""Samplers based on the `pocomc` inference engine."""

import torch
import pocomc
import numpy as np
from sunbird.inference.base import BaseSampler


class PocoMCSampler(BaseSampler):
    """PoCoMC sampler wrapper with optional ellipsoid prior support."""

    def __init__(self, **kwargs):
        """Initialize the PoCoMC sampler wrapper."""
        super().__init__(**kwargs)

    def fill_params(self, theta):
        """Fill a parameter vector to include fixed parameters.

        Args:
            theta: Free parameter vector.

        Returns:
            Filled parameter vector with fixed values inserted.
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
        """Fill a batch of parameter vectors to include fixed parameters.

        Args:
            thetas: Batch of free parameter vectors.

        Returns:
            Filled parameter array with fixed values inserted.
        """
        params = np.ones((len(thetas), len(self.priors.keys())))
        for i, theta in enumerate(thetas):
            params[i] = self.fill_params(theta)
        return params

    def get_model_prediction(self, theta):
        """Return the model prediction for the given parameters.

        Args:
            theta: Parameter vector or batch of vectors.

        Returns:
            Model prediction as a NumPy array.
        """
        pred = self.theory_model(x=theta, skip_output_inverse_transform=self.sample_in_transformed_space)
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        return pred

    def log_likelihood(self, theta):
        """Compute the log likelihood for a parameter vector or batch.

        Args:
            theta: Free parameter vector or batch of vectors.

        Returns:
            Log likelihood value(s).
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

    def __call__(
        self,
        vectorize=True,
        random_state=0,
        precondition=True,
        n_total=4096,
        progress=True,
        **kwargs,
    ):
        """Run the PoCoMC sampler.

        Args:
            vectorize: Vectorize the log likelihood call.
            random_state: Random seed for the sampler.
            precondition: If False, disable normalizing flow preconditioning.
            n_total: Total number of samples to draw.
            progress: Whether to display progress output.
            **kwargs: Additional arguments for `pocomc.Sampler`.
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
        """Return the posterior samples and derived quantities."""
        samples, weights, loglike, logprior = self.sampler.posterior()
        logz, logz_err = self.sampler.evidence()
        logposterior = loglike + logprior - logz
        return {'samples': samples, 'weights': weights, 'log_likelihood': loglike,
                'log_prior': logprior, 'log_posterior': logposterior}

    def evidence(self):
        """Return the evidence estimate and its error."""
        return self.sampler.evidence()


class PocoMCPriorSampler(PocoMCSampler):
    """PoCoMC sampler that returns a flat likelihood over the prior."""
    def __init__(
        self,
        observation=None,
        precision_matrix=None,
        theory_model=None,
        **kwargs,
    ):
        """Initialize a prior-only sampler."""
        super().__init__(observation, precision_matrix, theory_model, **kwargs)

    def log_likelihood(self, theta):
        """Return a flat log likelihood with optional ellipsoid term."""
        batch = len(theta.shape) > 1
        params = self.fill_params_batch(theta) if batch else self.fill_params(theta)
        if batch:
            logl = np.ones(len(theta))
            if self.ellipsoid:
                logl += np.asarray([self.abacus_ellipsoid.log_likelihood(params[i, :8]) for i in range(len(theta))])
        else:
            logl  = 1.0 
            if self.ellipsoid:
                logl += self.abacus_ellipsoid.log_likelihood(params[:8])
        return logl



if __name__ == "__main__":
    PocoMCSampler()