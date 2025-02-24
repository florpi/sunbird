from typing import Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro import infer
from functools import partial
from numpyro import distributions as dist
from jax import random
import time


class HMCSampler:
    def __init__(
        self,
        observation,
        precision_matrix,
        nn_theory_model,
        nn_parameters,
        priors,
        ranges,
        labels: Dict[str, str] = {},
        fixed_parameters: Dict[str, float] = {},
        model_filters: Dict = {},
    ):
        self.nn_theory_model = nn_theory_model
        self.nn_parameters = nn_parameters
        self.fixed_parameters = fixed_parameters
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
        self.model_filters = model_filters


    def sample_prior(
        self,
    ) -> Dict[str, float]:
        """Sample a set of parameters from the prior

        Returns:
            Dict: dictionary of parameters
        """
        t0 = time.time()
        x = jnp.ones(len(self.priors.keys()))
        for i, param in enumerate(self.priors.keys()):
            if self.fixed_parameters is None or param not in self.fixed_parameters.keys():
                x = x.at[i].set(
                    numpyro.sample(
                        param,
                        self.priors[param],
                    )
                )
            else:
                x = x.at[i].set(
                    numpyro.deterministic(param, self.fixed_parameters[param])
                )
        print(f'Sampling prior took {time.time() - t0:.2f} s')
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
        t0 = time.time()
        x = self.sample_prior()
        if hasattr(self.nn_theory_model, '__iter__'):
            prediction = []
            for model, params, filters in zip(self.nn_theory_model, self.nn_parameters, self.model_filters):
                pred, _ = model.apply(
                    params,
                    x,
                    filters=filters,
                )
                prediction.append(pred)
            prediction = jnp.concatenate(prediction)
        else:
            prediction, _  = self.nn_theory_model.apply(
                self.nn_parameters,
                x,
                filters=self.model_filters,
            )
        numpyro.sample(
            "y", dist.MultivariateNormal(prediction, precision_matrix=self.precision_matrix), obs=y
        )
        print(f'likelihood evaluation took {time.time() - t0:.2f} s')

    def __call__(
        self,
        kernel: str = "NUTS",
        num_warmup: int = 500,
        num_samples: int = 4000,
        num_chains: int = 1,
        save_fn: Optional[str] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        """Run the HMC inference

        Args:
            kernel (str, optional): kernel used for HMC. Defaults to "NUTS".
            num_warmup (int, optional): number of warmup steps. Defaults to 500.
            num_samples (int, optional): numper of samples. Defaults to 4000.
        """
        kernel = getattr(infer, kernel)(self.model, init_strategy=infer.init_to_mean, **kwargs)
        mcmc = infer.MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
        )
        rng_key = random.PRNGKey(0)
        mcmc.run(
            rng_key,
            y=self.observation,
            extra_fields=['potential_energy'],
        )
        mcmc.print_summary()
        results = mcmc.get_samples()
        if save_fn is not None:
            self.save_results(results, save_fn, metadata)
        return results
    
    def save_results(self, results, save_fn, metadata=None):
        samples = np.stack(list(results.values()), axis=0)
        idx = [list(results.keys()).index(param) for param in self.fixed_parameters]
        samples = np.delete(samples, idx, axis=0)
        names = np.delete(list(results.keys()), idx)
        cout = { 'samples': samples.T,
            'weights': np.ones(samples.shape[-1]),
            'param_ranges': self.ranges,
            'param_names': names,
            'param_labels': self.labels,
        }
        if metadata:
            cout.update(metadata)
        print(f'Saving results to {save_fn}')
        np.save(save_fn, cout)
        return
