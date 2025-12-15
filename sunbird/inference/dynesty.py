import dill
import torch
import numpy as np
import pandas as pd
import dynesty.utils
from multiprocessing import Pool
from typing import Dict, Optional
from dynesty import DynamicNestedSampler

dynesty.utils.pickle_module = dill


class DynestySampler():
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

    def get_log_likelihood(self, theta):
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

    def get_prior_from_cube(self, cube: np.array) -> np.array:
        """Transform a cube of uniform priors into the desired distribution

        Args:
            cube (np.array): uniform cube

        Returns:
            np.array: prior
        """
        transformed_cube = np.array(cube)
        itheta = 0
        for param in self.priors.keys():
            if param not in self.fixed_parameters.keys():
                transformed_cube[itheta] = self.priors[param].ppf(cube[itheta])
                itheta += 1
            else:
                continue
        return transformed_cube

    def __call__(
        self,
        nlive: int = 1000,
        dlogz: float = 0.01,
        save_fn: Optional[str] = None,
        nthreads=1,
    ):
        """Run nested sampling

        Args:
            num_live_points (int, optional): number of live points. Defaults to 500.
            dlogz (float, optional): allowed error on evidence. Defaults to 0.01.
            max_iterations (int, optional): maximum number of iterations. Defaults to 50_000.
            max_calls (int, optional): maximum number of calls. Defaults to 1_000_000.
        """
        if nthreads > 1:
            with Pool(nthreads) as pool:
                sampler = DynamicNestedSampler(
                    self.get_log_likelihood,
                    self.get_prior_from_cube,
                    ndim=self.ndim,
                    pool=pool,
                    queue_size=nthreads,
                    use_pool={'prior_transform': False},
                    bound='multi',
                )
                sampler.run_nested(
                    dlogz_init=dlogz,
                    nlive_init=nlive,
                )
        else:
            sampler = DynamicNestedSampler(
                self.get_log_likelihood,
                self.get_prior_from_cube,
                ndim=self.ndim,
                queue_size=6,
                bound='multi',
            )
            sampler.run_nested(
                dlogz_init=dlogz,
                nlive_init=nlive,
            )
        results = sampler.results
        if save_fn:
            self.save_results(results, save_fn)

    def save_results(self, results: Dict, save_fn: str):
        """Store inference results

        Args:
            results (Dict): dictionary with chain and summary statistics
        """
        df = self.convert_results_to_df(results=results)
        df.to_csv(save_fn, index=False)

    def convert_results_to_df(self, results: Dict) -> pd.DataFrame:
        """Convert dynesty results to pandas dataframe

        Args:
            results (Dict): dynesty results

        Returns:
            pd.DataFrame: summarised results
        """
        log_like = results.logl
        log_weights = results.logwt
        log_evidence = results.logz
        log_evidence_err = results.logzerr
        samples = results.samples
        df = pd.DataFrame(
            {
                "log_likelihood": log_like,
                "log_weights": log_weights,
                "log_evidence": log_evidence,
                "log_evidence_err": log_evidence_err,
            }
        )
        i = 0
        for param in self.priors.keys():
            if param not in self.fixed_parameters.keys():
                df[param] = samples[:, i]
                i += 1
            else:
                continue
        return df

    def get_results(
        self,
    ) -> pd.DataFrame:
        """
        Read results from file
        """
        return pd.read_csv(self.output_dir / "results.csv")
