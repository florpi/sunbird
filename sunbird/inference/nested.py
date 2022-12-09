from sunbird.inference import Inference
import json
import numpy as np
import pandas as pd
from dynesty import NestedSampler


class Nested(Inference):
    def get_prior_from_cube(self, cube):
        transformed_cube = np.array(cube)
        for n, param in enumerate(self.param_names):
            transformed_cube[n] = self.priors[param].ppf(cube[n])
        return transformed_cube

    def get_loglikelihood_for_params(self, params):
        prediction = self.get_model_prediction(params)
        return self.get_loglikelihood_for_prediction(prediction=prediction)

    def __call__(
        self,
        num_live_points=1000,
        dlogz=0.01,
        max_iterations=50_000,
        max_calls=1_000_000,
    ):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sampler = NestedSampler(
            self.get_loglikelihood_for_params,
            self.get_prior_from_cube,
            ndim=self.n_dim,
            nlive=num_live_points,
        )
        sampler.run_nested(
            checkpoint_file=str(self.output_dir / "dynasty.save"),
            dlogz=dlogz,
            maxiter=max_iterations,
            maxcall=max_calls,
        )
        results = sampler.results
        self.store_results(results)

    def store_results(self, results):
        df = self.convert_results_to_df(results=results)
        df.to_csv(self.output_dir / "results.csv", index=False)

    def convert_results_to_df(self, results):
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
        for i, param in enumerate(self.priors):
            df[param] = samples[:, i]
        return df

    def get_results(
        self,
    ):
        return pd.read_csv(self.output_dir / "results.csv")
