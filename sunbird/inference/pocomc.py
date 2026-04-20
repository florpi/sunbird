"""Samplers based on the `pocomc` inference engine."""

import torch
import pocomc
import numpy as np
from sunbird.inference.base import BaseSampler


class PocoMCSampler(BaseSampler):
    """PoCoMC sampler wrapper with optional ellipsoid prior support."""

    def __init__(self, number_density_model=None, target_density=None, **kwargs):
        """Initialize the PoCoMC sampler wrapper.
        
        Args:
            number_density_model: Optional model that predicts number density from HOD parameters.
            target_density: Optional minimum target number density threshold.
            **kwargs: Additional arguments passed to BaseSampler.
        """
        super().__init__(**kwargs)
        self.number_density_model = number_density_model
        self.target_density = target_density

        if self.number_density_model is not None and self.target_density is not None:
            self.logger.info(f"Applying number density constraint: predicted density >= {self.target_density}")

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

    def params_to_dict(self, params):
        """Convert parameter array to dictionary.

        Args:
            params: Full parameter array (1D) or batch of arrays (2D).

        Returns:
            Dictionary mapping parameter names to values or arrays of values.
        """
        param_names = list(self.priors.keys())
        params = np.asarray(params)

        if params.ndim == 1:
            # Single parameter vector
            return {name: np.array([params[i]]) for i, name in enumerate(param_names)}
        else:
            # Batch of parameter vectors
            return {name: params[:, i] for i, name in enumerate(param_names)}

    def extract_cosmo_params(self, params):
        """Extract cosmological parameters for ellipsoid prior.

        Args:
            params: Full parameter array or dictionary.

        Returns:
            Array of cosmological parameters in the correct order.
        """
        cosmo_param_names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld']
        param_names = list(self.priors.keys())
        
        if isinstance(params, dict):
            # Extract from dictionary
            return np.array([params[name] for name in cosmo_param_names if name in params])
        else:
            # Extract from array by finding indices
            params = np.asarray(params)
            indices = [param_names.index(name) for name in cosmo_param_names if name in param_names]
            if params.ndim == 1:
                return params[indices]
            else:
                return params[:, indices]

    def extract_hod_params(self, params):
        """Extract HOD parameters for number density calculation.

        Args:
            params: Full parameter array or dictionary.

        Returns:
            Dictionary or array of HOD parameters (excluding cosmological parameters).
        """
        cosmo_param_names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld']
        param_names = list(self.priors.keys())
        hod_param_names = [name for name in param_names if name not in cosmo_param_names]
        
        if isinstance(params, dict):
            # Return dictionary of HOD parameters
            return {name: params[name] for name in hod_param_names if name in params}
        else:
            # Extract from array by finding indices
            params = np.asarray(params)
            indices = [param_names.index(name) for name in hod_param_names]
            if params.ndim == 1:
                return params[indices]
            else:
                return params[:, indices]

    def get_model_prediction(self, theta):
        """Return the model prediction for the given parameters.

        Args:
            theta: Parameter vector or batch of vectors.

        Returns:
            Model prediction as a NumPy array.
        """
        if not hasattr(self, '_use_dict_input'):
            # First call - determine which input format the theory_model accepts
            try:
                theta_dict = self.params_to_dict(theta)
                pred = self.theory_model(x=theta_dict)
                self._use_dict_input = True  # Cache: dict input works
            except (TypeError, ValueError, KeyError):
                # Fallback to array input if dict not supported
                pred = self.theory_model(x=theta)
                self._use_dict_input = False  # Cache: use array input
        else:
            # Subsequent calls - use cached input format
            if self._use_dict_input:
                theta_dict = self.params_to_dict(theta)
                pred = self.theory_model(x=theta_dict)
            else:
                pred = self.theory_model(x=theta)
        
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        return pred

    def _check_density_constraint(self, params, batch=False):
        """Check if parameters satisfy the number density constraint.

        Args:
            params: Full parameter array or batch.
            batch: Whether this is a batch of parameters.

        Returns:
            For single sample: True if constraint satisfied, False otherwise.
            For batch: Boolean mask where True means constraint is satisfied.
        """
        if self.number_density_model is None or self.target_density is None:
            return True if not batch else np.ones(len(params), dtype=bool)
        
        with torch.no_grad():
            predicted_density = self.number_density_model.get_prediction(torch.Tensor(params))
            predicted_density = predicted_density.numpy()

        # Flatten the prediction to 1D to handle different output shapes
        predicted_density = predicted_density.ravel()
        
        if batch:
            return predicted_density >= self.target_density
        else:
            return predicted_density[0] >= self.target_density

    def _compute_data_log_likelihood(self, params, valid_mask=None):
        """Compute the data log likelihood term.

        Args:
            params: Full parameter array or batch.
            valid_mask: Optional boolean mask for batch processing (only compute for True entries).

        Returns:
            Log likelihood value(s) from data comparison.
        """
        prediction = self.get_model_prediction(params)
        diff = self.observation - prediction
        
        batch = len(params.shape) > 1
        if batch:
            if valid_mask is not None:
                # Only compute for valid samples
                logl = np.full(len(params), -np.inf)
                valid_indices = np.where(valid_mask)[0]
                logl[valid_indices] = np.asarray([-0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in valid_indices])
            else:
                logl = np.asarray([-0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(params))])
        else:
            logl = -0.5 * diff @ self.precision_matrix @ diff.T
        
        return logl

    def _apply_ellipsoid_prior(self, params, logl, valid_mask=None):
        """Apply the ellipsoid prior to the log likelihood.

        Args:
            params: Full parameter array or batch.
            logl: Current log likelihood value(s).
            valid_mask: Optional boolean mask for batch processing.

        Returns:
            Log likelihood with ellipsoid prior added.
        """
        if not self.ellipsoid:
            return logl
        
        cosmo_params = self.extract_cosmo_params(params)
        batch = len(params.shape) > 1
        
        if batch:
            ellipsoid_logl = np.asarray([self.abacus_ellipsoid.log_likelihood(cosmo_params[i]) for i in range(len(params))])
            if valid_mask is not None:
                # Only add to valid samples
                logl[valid_mask] += ellipsoid_logl[valid_mask]
            else:
                logl += ellipsoid_logl
        else:
            logl += self.abacus_ellipsoid.log_likelihood(cosmo_params)
        
        return logl

    def log_likelihood(self, theta):
        """Compute the log likelihood for a parameter vector or batch.

        Args:
            theta: Free parameter vector or batch of vectors.

        Returns:
            Log likelihood value(s).
        """
        batch = len(theta.shape) > 1
        params = self.fill_params_batch(theta) if batch else self.fill_params(theta)
        
        # Check density constraint
        density_valid = self._check_density_constraint(params, batch=batch)
        
        if batch:
            # Early exit if all samples fail density constraint
            if not np.any(density_valid):
                return np.full(len(theta), -np.inf)
            # Compute data likelihood (only for valid samples if constraint applied)
            valid_mask = density_valid if (self.number_density_model is not None) else None
            logl = self._compute_data_log_likelihood(params, valid_mask=valid_mask)
            # Apply ellipsoid prior
            logl = self._apply_ellipsoid_prior(params, logl, valid_mask=valid_mask)
        else:
            # Early exit if density constraint fails
            if not density_valid:
                return -np.inf
            # Compute data likelihood and apply ellipsoid prior
            logl = self._compute_data_log_likelihood(params)
            logl = self._apply_ellipsoid_prior(params, logl)
        
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
        """Return a flat log likelihood with optional density and ellipsoid constraints."""
        batch = len(theta.shape) > 1
        params = self.fill_params_batch(theta) if batch else self.fill_params(theta)
        
        # Check density constraint
        density_valid = self._check_density_constraint(params, batch=batch)
        
        if batch:
            # Initialize with flat likelihood
            logl = np.ones(len(theta))
            # Set invalid samples to -inf
            logl[~density_valid] = -np.inf
            # Apply ellipsoid prior only to valid samples
            valid_mask = density_valid if (self.number_density_model is not None) else None
            logl = self._apply_ellipsoid_prior(params, logl, valid_mask=valid_mask)
        else:
            # Early exit if density constraint fails
            if not density_valid:
                return -np.inf
            # Start with flat likelihood and apply ellipsoid prior
            logl = 1.0
            logl = self._apply_ellipsoid_prior(params, logl)
        
        return logl



if __name__ == "__main__":
    PocoMCSampler()