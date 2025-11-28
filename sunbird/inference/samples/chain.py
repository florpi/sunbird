from pathlib import Path
from getdist import plots, MCSamples
import numpy as np
from .base import Samples
import logging


class Chain(Samples):
    """
    Class to read chains generated with sunbird's inference module.
    """
    def __init__(self, data=None):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.samples = data['samples']
        self.weights = data['weights']
        self.names = data['names']
        self.fixed_params = data['fixed_parameters']
        self.ranges = data['ranges']
        self.loglike = data['log_likelihood']
        self.max_likelihood = data['max_likelihood']
        self.max_posterior = data['max_posterior']
        self.markers = data['markers']
        self.labels = [data['labels'][n] for n in data['names']]
        self.bestfit = self.get_bestfit()

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

    def get_bestfit(self, dist='posterior'):
        """
        Get the maximum a posterior point from a chain.
        """
        max_point = self.max_posterior if dist == 'posterior' else self.max_likelihood
        best_fit = {key: val for key, val in zip(self.names, max_point)}
        best_fit.update(self.fixed_params)
        return best_fit

    def plot_triangle(self, save_fn=None, thin=1, add_bestfit=False, show=False, **kwargs):
        """Plot triangle plot
        """
        import matplotlib.pyplot as plt
        names = self.names
        labels = [label.strip('$') for label in self.labels]
        samples = MCSamples(samples=self.samples, weights=self.weights, names=names,
                            loglikes=self.loglike, labels=labels, ranges=self.ranges)
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, **kwargs)
        maxl = self.samples[self.loglike.argmax()]
        if add_bestfit:
            params = kwargs['params'] if 'params' in kwargs else names
            ndim = len(params)
            finished = []
            ax_idx = 0
            for i, param1 in enumerate(params):
                for j, param2 in enumerate(params[::-1]):
                    if param2 in finished: continue
                    if param1 != param2:
                        g.fig.axes[ax_idx].plot(maxl[names.index(param1)], maxl[names.index(param2)],
                                                marker='*', ms=10.0, color='k', mew=1.0, mfc='w')
                    ax_idx += 1
                finished.append(param1)
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_trace(self, save_fn=None, thin=1, show=False):
        """Parameter trace plot
        """
        names = self.names
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 2*len(names)))
        for i, name in enumerate(names):
            ax[i].plot(self.samples[:, i])
            ax[i].set_ylabel(self.labels[i])
        ax[i].set_xlabel('Iteration')
        plt.tight_layout()
        if save_fn:
            self.logger.info(f'Saving {save_fn}')
            plt.savefig(save_fn, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()