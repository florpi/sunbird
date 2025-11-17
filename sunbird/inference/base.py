from sunbird.inference.priors import AbacusSummitEllipsoid
from typing import Dict, Optional
from collections.abc import Iterable
from getdist import plots, MCSamples
import logging


class BaseSampler:
    def __init__(self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        ranges: Optional[Dict[str, tuple]] = {},
        labels: Dict[str, str] = {},
        fixed_parameters: Dict[str, float] = {},
        slice_filters: Dict = {},
        select_filters: Dict = {},
        coordinates: list = [],
        ellipsoid: bool = False,
        markers: dict = {},
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.theory_model = theory_model
        if fixed_parameters is None:
            fixed_parameters = {}
        self.fixed_parameters = fixed_parameters
        self.observation = observation
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.precision_matrix = precision_matrix
        self.ellipsoid = ellipsoid
        self.markers = markers
        if self.ellipsoid:
            self.abacus_ellipsoid = AbacusSummitEllipsoid()
        self.ndim = len(self.priors.keys()) - len(self.fixed_parameters.keys())
        self.logger.info(f'Free parameters: {[key for key in priors.keys() if key not in fixed_parameters.keys()]}')
        self.logger.info(f'Fixed parameters: {[key for key in priors.keys() if key in fixed_parameters.keys()]}')

    def save_chain(self, save_fn, metadata=None):
        """Save the chain to a file
        """
        import numpy as np
        data = self.get_chain(flat=True)
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        dof = len(self.observation) - self.ndim
        maxl = data['samples'][data['log_likelihood'].argmax()]
        maxp = data['samples'][data['log_posterior'].argmax()]
        cout = {**data,
            'ranges': self.ranges,
            'names': names,
            'labels': self.labels,
            'fixed_parameters': self.fixed_parameters,
            'dof': dof,
            'max_likelihood': maxl,
            'max_posterior': maxp,
        }
        if metadata:
            for key, val in metadata.items():
                cout[key] = val
        if hasattr(self, 'evidence'):
            logz, logz_error = self.evidence()
            cout['logz'] = logz
            cout['log_zerror'] = logz_error
        self.logger.info(f'Saving {save_fn}')
        np.save(save_fn, cout)

    def save_table(self, save_fn):
        from tabulate import tabulate
        chain = self.get_chain(flat=True)
        maxp = chain['samples'][chain['log_posterior'].argmax()]
        mean = chain['samples'].mean(axis=0)
        std = chain['samples'].std(axis=0)
        dof = len(self.observation) - self.ndim
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        headers = ['parameter', 'MAP', 'mean', 'std']
        has_evidence = hasattr(self, 'evidence')
        if has_evidence:
            logz, logz_err = self.evidence()
        table = []
        for i, name in enumerate(names):
            row = [name, f"{maxp[i]:.4f}", f"{mean[i]:.4f}", f"{std[i]:.4f}"]
            table.append(row)
        with open(save_fn, 'w') as f:
            self.logger.info(f'Saving {save_fn}')
            f.write(tabulate(table, tablefmt='pretty', headers=headers))
            chi2 = -2 * chain['log_likelihood'].max()
            f.write(f"\n\nchi2/dof: {chi2:.2f}/{dof}")
            if hasattr(self, 'evidence'):
                logz, logz_err = self.evidence()
                f.write("\n\nEvidence calculation:\n")
                f.write(f"log(Z) = {logz:.4f} +- {logz_err:.4f}\n")
