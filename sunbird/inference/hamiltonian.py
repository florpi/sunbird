from typing import Dict
import jax
import jax.numpy as jnp
import numpyro
from numpyro import infer
from numpyro import distributions as dist
from jax import random

class HMC:
    def __init__(
        self,
        observation,
        precision_matrix,
        nn_theory_model,
        nn_parameters,
        priors,
    ):
        self.nn_theory_model = nn_theory_model
        self.nn_parameters = nn_parameters 
        self.observation = observation
        self.priors = priors
        self.precision_matrix = precision_matrix 


    def sample_prior(
        self,
    ) -> Dict[str, float]:
        """Sample a set of parameters from the prior

        Returns:
            Dict: dictionary of parameters
        """

        x = jnp.ones(len(self.priors.keys()))
        for i, param in enumerate(self.priors.keys()):
            x = x.at[i].set(
                numpyro.sample(
                    param,
                    self.priors[param],
                )
            )
        return x

    def test_sample_prior(
        self,
        key,
    ) -> Dict[str, float]:
        """Sample a set of parameters from the prior

        Returns:
            Dict: dictionary of parameters
        """

        x = jnp.ones(len(self.priors.keys()))
        for i, param in enumerate(self.priors.keys()):
            key, subkey = jax.random.split(key)
            x = x.at[i].set(
                numpyro.sample(
                    param,
                    self.priors[param],
                    rng_key=subkey
                )
            )
        return x

    def sanity_check_prior(self, n_samples=10, seed=0):
        key = random.PRNGKey(seed)
        predictions = []
        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            x = self.test_sample_prior(key=subkey)
            prediction, _  = self.nn_theory_model.apply(
                self.nn_parameters,
                x,
            )
            predictions.append(prediction)
        return predictions

    def model(
        self,
        y: jnp.array,
    ):
        """Likelihood evaluation for the HMC inference

        Args:
            y (np.array): array with observation
        """
        x = self.sample_prior()
        prediction, _  = self.nn_theory_model.apply(
            self.nn_parameters,
            x,
        )
        numpyro.sample(
            "y", dist.MultivariateNormal(prediction, precision_matrix=self.precision_matrix), obs=y
        )


    def __call__(
        self,
        kernel: str = "NUTS",
        num_warmup: int = 100,
        num_samples: int = 1000,
    ):
        """Run the HMC inference

        Args:
            kernel (str, optional): kernel used for HMC. Defaults to "NUTS".
            num_warmup (int, optional): number of warmup steps. Defaults to 500.
            num_samples (int, optional): numper of samples. Defaults to 1000.
        """
        kernel = getattr(infer, kernel)(self.model)
        mcmc = infer.MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        rng_key = random.PRNGKey(0)
        mcmc.run(
            rng_key,
            y=self.observation,
            extra_fields=['potential_energy'],
        )
        mcmc.print_summary()
        results = mcmc.get_samples()
        return results