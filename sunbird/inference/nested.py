from sunbird.inference import Inference
import numpy as np
from ultranest import ReactiveNestedSampler
import ultranest.stepsampler


class Nested(Inference):
    def get_prior_from_cube(self, cube):
        transformed_cube = np.zeros_like(cube)
        for n, param in enumerate(self.param_names):
            transformed_cube[:, n] = self.priors[param].ppf(cube[:, n])
        return transformed_cube

    def get_loglikelihood_for_params(self, params):
        prediction = self.get_model_prediction(params)
        loglike = self.get_loglikelihood_for_prediction(prediction=prediction)
        return np.atleast_1d(loglike)

    def __call__(self, log_dir, slice_steps, num_live_points):
        sampler = ReactiveNestedSampler(
            self.param_names,
            self.get_loglikelihood_for_params,
            log_dir=log_dir,
            vectorized=True,
            transform=self.get_prior_from_cube,
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=slice_steps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )
        sampler.run(
            dlogz=0.5 + 0.1 * len(self.param_names),
            max_num_improvement_loops=3,
            min_num_live_points=num_live_points,
        )
        sampler.print_results()
        sampler.plot()

    def get_results(
        self,
    ):
        chain = np.loadtxt(
            self.inference_config["outputfiles_basename"] + f"smin{self.s_min:.2f}.txt"
        )
        df = pd.DataFrame(
            {
                "log_likelihood": chain[:, 1],
                "weights": chain[:, 0],
            }
        )
        for i, param in enumerate(self.priors):
            df[param] = chain[:, 2 + i]
        return df
