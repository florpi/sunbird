from typing import Dict
from functools import partial
import pandas as pd
import jax
from jax import random
import jax.numpy as np
import numpyro
from numpyro import infer
from numpyro import distributions as dist
from sunbird.inference import Inference


class HMC(Inference):
    def sample_prior(
        self,
    ) -> Dict[str, float]:
        """Sample a set of parameters from the prior

        Returns:
            Dict: dictionary of parameters
        """
        x = np.ones(len(self.theory_model.input_names))
        for i, param in enumerate(self.theory_model.input_names):
            if param in self.priors:
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
        return x


    def model(
        self,
        y: np.array,
    ):
        """Likelihood evaluation for the HMC inference

        Args:
            y (np.array): array with observation
        """
        x = self.sample_prior()
        prediction, predicted_uncertainty = self.theory_model.forward(
            x,
            select_filters=self.select_filters,
            slice_filters=self.slice_filters,
        )
        if self.add_predicted_uncertainty:
            covariance_matrix = self.covariance_matrix + np.diag(predicted_uncertainty**2)
        else:
            covariance_matrix = self.covariance_matrix
        numpyro.sample(
            "y", dist.MultivariateNormal(prediction, covariance_matrix), obs=y
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
        )
        mcmc.print_summary()
        results = mcmc.get_samples()
        self.store_results(results)

    def store_results(self, results):
        """Store results in a csv file

        Args:
            results: chain of samples
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_dict(results)
        df.to_csv(self.output_dir / "results.csv", index=False)
