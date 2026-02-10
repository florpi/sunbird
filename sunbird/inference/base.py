"""Base classes and utilities for inference samplers."""

import logging
import numpy as np
from tabulate import tabulate
from typing import Dict, Optional
from sunbird.inference.priors import AbacusSummitEllipsoid

class BaseSampler:
    """Base class for inference samplers.

    Handles parameter bookkeeping, optional transformed-space sampling, and
    convenience utilities for saving chains and summary tables.
    """

    def __init__(
        self,
        observation,
        precision_matrix,
        theory_model,
        priors,
        ranges: Optional[Dict[str, tuple]] = {},
        labels: Dict[str, str] = {},
        fixed_parameters: Dict[str, float] = {},
        ellipsoid: bool = False,
        markers: dict = {},
        sample_in_transformed_space: bool = False,
        **kwargs,
    ):
        """Initialize the sampler base.

        Args:
            observation: Observed data vector.
            precision_matrix: Inverse covariance matrix.
            theory_model: Callable model that maps parameters to predictions.
            priors: Mapping of parameter names to prior objects.
            ranges: Optional plotting or reporting ranges by parameter.
            labels: Optional labels by parameter.
            fixed_parameters: Mapping of parameter names to fixed values.
            ellipsoid: Whether to include the AbacusSummit ellipsoid prior.
            markers: Optional marker styling for plots.
            sample_in_transformed_space: If True, use transformed outputs.
            **kwargs: Extra arguments for subclasses.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.theory_model = theory_model
        if fixed_parameters is None:
            fixed_parameters = {}
        self.fixed_parameters = fixed_parameters
        self.priors = priors
        self.ranges = ranges
        self.labels = labels
        self.ellipsoid = ellipsoid
        self.markers = markers
        self.sample_in_transformed_space = sample_in_transformed_space
        
        # Handle transformation of observations and covariance
        if sample_in_transformed_space:
            # Validate that the observable has an output transform
            if not hasattr(theory_model.__self__.model, 'transform_output'):
                raise ValueError('Cannot sample in transformed space: observable does not have a transform_output. '
                                'Either set sample_in_transformed_space=False or use an observable with transform_output.')
            
            # Check if transform_output is valid (not None or empty list)
            transform = theory_model.__self__.model.transform_output
            if transform is None:
                raise ValueError('Cannot sample in transformed space: transform_output is None. '
                                'Either set sample_in_transformed_space=False or use an observable with transform_output.')
            
            # For combined observables, transform_output is a list
            if isinstance(transform, list):
                if all(t is None for t in transform):
                    raise ValueError('Cannot sample in transformed space: all transforms in combined observable are None. '
                                    'Either set sample_in_transformed_space=False or use observables with transform_output.')

            self.logger.warning('Sampling in transformed space (skip_output_inverse_transform=True). '
                                'Ensure observations and covariance matrix are also transformed to match!')
                
        self.observation = observation
        self.precision_matrix = precision_matrix
            
        if self.ellipsoid:
            self.abacus_ellipsoid = AbacusSummitEllipsoid()
            
        self.ndim = len(self.priors.keys()) - len(self.fixed_parameters.keys())
        self.logger.info(f'Free parameters: {[key for key in priors.keys() if key not in fixed_parameters.keys()]}')
        self.logger.info(f'Fixed parameters: {[key for key in priors.keys() if key in fixed_parameters.keys()]}')

    def save_chain(self, save_fn, metadata=None):
        """Save a chain dictionary to a NumPy file.

        Args:
            save_fn: Output filename for the NumPy archive.
            metadata: Optional extra metadata to include.
        """
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
        """Write a summary table with MAP/mean/std values.

        Args:
            save_fn: Output filename for the text table.
        """
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
