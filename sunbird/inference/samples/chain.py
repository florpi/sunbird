from pathlib import Path
from getdist import plots, MCSamples
import numpy as np


class Chain:
    """
    Class to read chains generated with sunbird's inference module.
    """
    def __init__(self, data=None):
        self.data = data
        self.samples = data['samples']
        self.weights = data['weights']
        self.names = data['names']
        self.fixed_params = data['fixed_parameters']
        self.ranges = data['ranges']
        self.markers = data['markers']
        self.labels = [data['labels'][n] for n in data['names']]

    @classmethod
    def load(self, filename):
        """
        Load a chain from a file.

        Args:
            filename (str): path to the chain file.

        Returns:
            dict: dictionary containing the chain.
        """
        data = np.load(filename, allow_pickle=True).item()
        return Chain(data)

    @classmethod
    def to_getdist(self, chain, add_derived=False, **kwargs):
        """
        Convert data to a GetDist MCSamples object.

        Args:
            data (dict): dictionary containing the chain data.

        Returns:
            MCSamples: GetDist MCSamples object.
        """
        mcsamples = MCSamples(
            samples=chain.samples,
            weights=chain.weights,
            names=chain.names,
            ranges=chain.ranges,
            labels=chain.labels,
            **kwargs,
        )
        if add_derived:
            mcsamples = chain.add_derived_params(mcsamples)
        return mcsamples

    def add_derived_params(self, mcsamples):
        """
        Add derived cosmological parameters to MCSamples object.

        Args:
            mcsamples (MCSamples): GetDist MCSamples object.

        Returns:
            MCSamples: GetDist MCSamples object with derived parameters.
        """
        from sunbird.cosmology.growth_rate import Growth
        varied_params = self.names
        fixed_params = self.fixed_params
        growth = Growth(emulate=True,)
        cosmo_labels = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'N_ur', 'w0_fld', 'wa_fld']
        cosmo_params = {}
        for label in cosmo_labels:
            if label in varied_params:
                cosmo_params[label] = mcsamples[label]
            else:
                cosmo_params[label] = np.ones_like(mcsamples[varied_params[0]]) * fixed_params[label]
        cosmo_params['sigma8'] = cosmo_params.pop('sigma8_m')
        h = growth.get_emulated_h(**cosmo_params)
        Omega_m = growth.Omega_m0(mcsamples['omega_cdm'], mcsamples['omega_b'], h.flatten())
        mcsamples.addDerived(h, name='h', label='h')
        mcsamples.addDerived(Omega_m, name=r'Omega_m', label=r'\Omega_{\rm m}')
        if 'zeff' in self.data:
            fsigma8 = growth.get_fsigma8(**cosmo_params, z=self.data['zeff'])
            mcsamples.addDerived(fsigma8, name=r'fsigma8', label=r'f\sigma_8')
        return mcsamples