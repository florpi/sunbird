import torch
import logging
import pybobyqa
import numpy as np
from desilike import mpi
from typing import Dict, Optional
from sunbird.data.data_utils import convert_to_summary


class BobyqaProfiler:
    """Bobyqa likelihood/posterior profiler class."""
    def __init__(
        self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        start: Optional[Dict[str, float]] = None,
        ranges: Optional[Dict[str, tuple]] = {},
        labels: Dict[str, str] = {},
        fixed_params: Dict[str, float] = {},
        slice_filters: Dict = {},
        select_filters: Dict = {},
        coordinates: list = [],
        mpicomm=None,
    ):
        self.theory_model = theory_model if isinstance(theory_model, list) else [theory_model]
        self.select_filters = select_filters if isinstance(select_filters, list) else [select_filters]
        self.slice_filters = slice_filters if isinstance(slice_filters, list) else [slice_filters]
        self.coordinates = coordinates if isinstance(coordinates, list) else [coordinates]
        self.fixed_params = fixed_params
        self.varied_params = [param for param in priors.keys() if param not in fixed_params.keys()]
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
        self.ndim = len(self.varied_params)

        if mpicomm is None:
            self.mpicomm = mpi.COMM_WORLD
        self.mpicomm = mpicomm

        if mpicomm.rank == 0:
            self.logger = logging.getLogger('BobyqaProfiler')
            self.logger.info('Initializing BobyqaProfiler.')
            self.logger.info(f'Free parameters: {self.varied_params}')
            self.logger.info(f'Fixed parameters: {[key for key in priors.keys() if key in fixed_params.keys()]}')

        self.bobyqa_params = {}
        self.bobyqa_params['name'] = self.varied_params

        self.start = start
        if start is None:
            self.start = self.get_start(limits=self.ranges)

    def get_start(self, limits=None):
        """Get the starting point for the optimizer

        Args:
            limits (dict, optional): parameter limits. Defaults to None,
            in which case we take the limits from the priors.

        Returns:
            list: starting point
        """
        if limits is None:
            limits = self.ranges
        start = [np.random.uniform(limits[key][0], limits[key][1])
                 for key in self.varied_params]
        return start

    def get_bounds(self, ranges):
        """Get bounds for the optimizer
        
        Args:
            ranges (dict): parameter ranges

        Returns:
            tuple: lower and upper bounds
        """
        lower = np.array([ranges[param][0] for param in self.varied_params])
        upper = np.array([ranges[param][1] for param in self.varied_params])
        return (lower, upper)

    def _minimize_one(self, start, max_calls=int(1e4), bounds=None):
        """Minimize the log likelihood function for one set of parameters

        Args:
            start (list): starting point
            max_calls (int, optional): maximum number of calls. Defaults to int(1e4).
            bounds (tuple, optional): lower and upper bounds. Defaults to None.

        Returns:
            dict: minimization instance from pybobyqa.solve
        """
        npt = int((self.ndim + 2) * (self.ndim + 1)/2)
        return pybobyqa.solve(
            objfun=self.log_likelihood,
            x0=start,
            bounds=bounds,
            npt=npt,
            scaling_within_bounds=True,
            seek_global_minimum=True,
            maxfun=max_calls,
            do_logging=False
        )
        
    def minimize(self, max_calls=int(1e4), save_fn=None):
        """Minimize the log likelihood function

        Args:
            max_calls (int, optional): maximum number of calls. Defaults to int(1e4).
            save_fn (str, optional): save filename. Defaults to None.

        Returns:
            dict: profile
        """
        bounds = self.get_bounds(self.ranges)
        start = self.start
        profile = {}
        minimize = self._minimize_one(start=start, max_calls=max_calls, bounds=bounds)
        all_minimize = self.mpicomm.gather(minimize, root=0)
        if self.mpicomm.rank == 0:
            best_idx = np.argmin([minimize.f for minimize in all_minimize])
            bestfit = {key: all_minimize[best_idx].x[i] for i, key in enumerate(self.varied_params)}
            chi2 = all_minimize[best_idx].f
            profile['bestfit'] = bestfit
            profile['chi2'] = chi2
            if save_fn is not None:
                np.save(save_fn, profile)
        self.profile = profile
        return self.profile

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
            if param not in self.fixed_params.keys():
                params[i] = theta[itheta]
                itheta += 1
            else:
                params[i] = self.fixed_params[param]
        return params

    def apply_model_filters(self, prediction, coordinates,
        select_filters, slice_filters, batch=False):
        """Apply filters to the model prediction.
        
        Args:
            prediction (np.array): model prediction
            coordinates (dict): coordinates of the model prediction
            select_filters (dict): select filters
            slice_filters (dict): slice filters
            
            Returns:
                np.array: filtered prediction
        """
        coords = coordinates.copy()
        if batch:
            coords_shape = tuple(len(v) for k, v in coords.copy().items())
            dimensions = ["batch"] + list(coords.keys())
            coords["batch"] = range(len(prediction))
            prediction = prediction.reshape((len(prediction), *coords_shape))
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=select_filters, slice_filters=slice_filters
            ).values.reshape(len(prediction), -1)
        else:
            coords_shape = tuple(len(v) for k, v in coords.items())
            prediction = prediction.reshape(coords_shape)
            dimensions = list(coords.keys())
            return convert_to_summary(
                data=prediction, dimensions=dimensions, coords=coords,
                select_filters=select_filters, slice_filters=slice_filters
            ).values.reshape(-1)

    def get_model_prediction(self, theta, batch=False):
        """Get model prediction

        Args:
            theta (np.array): input parameters

        Returns:
            np.array: model prediction
        """
        with torch.no_grad():
            prediction = []
            for i, model in enumerate(self.theory_model):
                pred = model.get_prediction(
                    x=torch.Tensor(theta),
                )
                if self.select_filters or self.slice_filters:
                    pred = self.apply_model_filters(
                        prediction=pred,
                        coordinates=self.coordinates[i],
                        select_filters=self.select_filters[i],
                        slice_filters=self.slice_filters[i],
                        batch=batch
                    )
                prediction.append(pred)
            prediction = np.concatenate(prediction, axis=-1)
            return prediction

    def log_likelihood(self, theta):
        """Log likelihood function

        Args:
            theta (np.array): input parameters

        Returns:
            float: log likelihood
        """
        theta = np.array(theta)
        params = self.fill_params_batch(theta) if len(theta.shape) > 1 else self.fill_params(theta)
        prediction = self.get_model_prediction(params)
        diff = self.observation - prediction
        logl = -0.5 * diff @ self.precision_matrix @ diff.T
        if len(theta.shape) > 1:
            return [0.5 * diff[i] @ self.precision_matrix @ diff[i].T for i in range(len(theta))] 
        return  0.5 * diff @ self.precision_matrix @ diff.T

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
            if param not in self.fixed_params.keys():
                dist = self.priors[param]
                lp[itheta] = dist.logpdf(theta[itheta])
                itheta += 1
            else:
                continue
        if any(~np.isfinite(lp)):
            return -np.inf
        return np.sum(lp)

    def log_posterior(self, theta):
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

    def save_table(self, save_fn):
        """
        Save the table of bestfit parameters
        
        Args:
            save_fn (str): save filename
        """
        if self.mpicomm.rank == 0:
            from tabulate import tabulate
            profile = self.profile
            maxl = profile['bestfit']
            chi2 = profile['chi2']
            dof = len(self.observation) - len(self.fixed_params)
            names = [param for param in self.priors.keys() if param not in self.fixed_params]
            chi2_header = f'chi2 / ({len(self.observation)} - {len(names)}) = {chi2:.2f} / {dof} = {chi2/dof:.2f}'
            headers = [chi2_header, 'max-like']
            table = []
            for name in names:
                table.append([name, f"{maxl[name]:.4f}"])
            with open(save_fn, 'w') as f:
                self.logger.info(f'Saving {save_fn}')
                f.write(tabulate(table, tablefmt='pretty', headers=headers))