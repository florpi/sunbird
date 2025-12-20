import torch
import pocomc
import numpy as np
from sunbird.inference.base import BaseSampler


class PocoMCSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        # pred = self.theory_model.get_prediction(x=theta)
        pred = self.theory_model(x=theta)
        # detach if using torch
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        return pred

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
        # detach if using torch
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().numpy()
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
        samples, weights, loglike, logprior = self.sampler.posterior()
        logz, logz_err = self.sampler.evidence()
        logposterior = loglike + logprior - logz
        return {'samples': samples, 'weights': weights, 'log_likelihood': loglike,
                'log_prior': logprior, 'log_posterior': logposterior}

    def evidence(self,):
        """Get the evidence from the sampler

        Returns:
            tuple: logz, logz_err
        """
        return self.sampler.evidence()


class PocoMCPriorSampler(PocoMCSampler):
    def __init__(
        self,
        observation=None,
        precision_matrix=None,
        theory_model=None,
        **kwargs,
    ):
        super().__init__(observation, precision_matrix, theory_model, **kwargs)

    def log_likelihood(self, theta):
        """Log likelihood function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log likelihood
        """
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