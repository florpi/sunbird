import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from getdist import plots, MCSamples
from sunbird.cosmology.growth_rate import Growth
from .base import Samples

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
    def load(self, filename: str):
        """
        Load a chain from a file.

        Parameters
        ----------
        filename : str
            Path to the chain file. The file should be a numpy .npy file.

        Returns
        -------
        dict: 
            Dictionary containing the chain.
        """
        data = np.load(filename, allow_pickle=True).item()
        return Chain(data)

    @classmethod
    def to_getdist(self, chain, add_derived: bool = False, **kwargs) -> MCSamples:
        """
        Convert data to a GetDist MCSamples object.

        Parameters
        ----------
        chain : Chain
            Chain object containing the samples and weights.
        add_derived : bool, optional
            Whether to add derived cosmological parameters, by default False.
        **kwargs:
            Additional arguments for MCSamples.

        Returns
        -------
        MCSamples: 
            GetDist MCSamples object.
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

    def add_derived_params(self, mcsamples: MCSamples) -> MCSamples:
        """
        Add derived cosmological parameters to MCSamples object.

        Parameters
        ----------
        mcsamples : MCSamples
            GetDist MCSamples object.

        Returns
        -------
        MCSamples: 
            GetDist MCSamples object with derived parameters.
        """
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

    def get_bestfit(self, dist: str = 'posterior') -> dict:
        """
        Get the maximum a posterior point from a chain.
        
        Parameters
        ----------
        dist : str, optional
            Distribution to use ('posterior' or 'likelihood'), by default 'posterior'.
        
        Returns
        -------
        dict:
            Dictionary of best-fit parameters.
        """
        max_point = self.max_posterior if dist == 'posterior' else self.max_likelihood
        best_fit = {key: val for key, val in zip(self.names, max_point)}
        best_fit.update(self.fixed_params)
        return best_fit
    
    def plot_trace(self, save_fn: str = None):
        """
        Parameter trace plot
        """
        # TODO: add option to select which parameters to plot
        names = self.names
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 2*len(names)))
        for i, name in enumerate(names):
            ax[i].plot(self.samples[:, i])
            ax[i].set_ylabel(self.labels[i])
        ax[i].set_xlabel('Iteration')
        plt.tight_layout()
        if save_fn:
            self.logger.info(f'Saving {save_fn}')
            plt.savefig(save_fn, bbox_inches='tight')
        return fig, ax
    
    def plot_triangle(
        self,
        *chains,
        save_fn: str = None,
        add_bestfit: bool = False,
        **kwargs, 
    ) -> plots.GetDistPlotter:
        """
        Plots the triangle plot of the chains loaded in the class, using the getdist package.

        Parameters
        ----------
        add_bestfit : bool, optional
            If True, will add the best fit point for each chain on the triangle plot.
            Defaults to False.
        chains : Chain
            Additional Chain objects to include in the triangle plot.
        save_fn : str, optional
            If provided, will save the figure to the given filename.
        **kwargs : dict, optional
            Additional keyword arguments to customize the plot. 
            See the getdist documentation for more options.

        Returns
        -------
        getdist.plots.GetDistPlotter
            The plotter object containing the triangle plot.
        """
        chains = [self] + list(chains)
        colors = kwargs.get('colors', ['k'] + [f'C{i}' for i in range(len(chains)-1)])
        label_dict = kwargs.get('label_dict', {}) 
        
        params = kwargs.pop('params', None)
        if params is not None:
            kwargs['params'] = [p for p in params if p in self.names] # Remove any parameter that is not in the chain (just in case)
    
        samples = []
        for chain in chains:
            chain.labels = [label.strip('$') for label in chain.labels] # getdist does not like $ in labels
            chain_label = label_dict.get(chain.data.get('label', ''), chain.data.get('label', None)) # The label for the chain in the triangle plot
            chain = self.to_getdist(chain, label=chain_label) # ensure it's a MCSamples object
            samples.append(chain)
        
        g = plots.get_subplot_plotter()
        g.triangle_plot(
            samples,
            line_args=[{'color': colors[i]} for i in range(len(samples))], # Ensure the histograms and contours have the same colors
            contour_colors=colors, 
            **kwargs,
        )
        if add_bestfit:
            for c, chain in enumerate(chains):
                names = chain.names # To get the right index later
                params = kwargs.get('params', names) # Will luckily already be in order because params will re-order the axes in the triangle plot 
                params = [p for p in params if p in names] # Remove any parameter that is not in the chain (just in case)
                maxl = chain.samples[chain.loglike.argmax()]
                finished = []
                ax_idx = 0
                for i, param1 in enumerate(params):
                    for j, param2 in enumerate(params[::-1]):
                        if param2 in finished: continue
                        if param1 != param2:
                            g.fig.axes[ax_idx].plot(maxl[names.index(param1)], maxl[names.index(param2)],
                                                    marker='*', ms=10.0, color=colors[c], mew=1.0, mfc='w')
                        ax_idx += 1
                    finished.append(param1)
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        return g

    def plot_map(self, *chains, percentile: int|list = 95, **kwargs):
        """
        Plots the Maximum A Posteriori (MAP) point, the mean and the 95% confidence interval (default) for each parameter in the chain.

        Parameters
        ----------
        percentile : int | list, optional
            The percentile to use for the error bars. 
            If an integer is provided, the error bars will both be the same and correspond to the given percentile.
            If a list is provided, the first element will be the lower percentile and the second element the upper percentile.
            Defaults to 95.
        chains : Chain
            Additional Chain objects to include in the plot.
        **kwargs : dict, optional
            Additional keyword arguments to customize the plots.
            Can contain the following keys:
            - colors: list of str, the colors to use for the chains. Defaults to ['k'] + [f'C{i}' for i in range(len(self.chains))].
            - label_dict: dict, a dictionary containing the labels to use for the chains. Defaults to {}.
            - params: list of str, the parameters to plot. Defaults to self.names.
            - markers: dict, a dictionary containing the value of vertical markers to plot for each parameter. Defaults to None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        chains = [self] + list(chains)
        percentile_list = [percentile, percentile] if isinstance(percentile, int) else percentile
        colors = kwargs.get('colors', ['k'] + [f'C{i}' for i in range(len(chains)-1)])
        label_dict = kwargs.get('label_dict', {}) 
        params = kwargs.get('params', self.names)
        names = [p for p in params if p in self.names]
        labels = {k: v for k, v in zip(self.names, self.labels)}
        labels = [labels[n] for n in names]
        chain_labels = [label_dict.get(chain.data.get('label', ''), chain.data.get('label', None)) for chain in chains] # replace label with actual name if provided
        
        fig, ax = plt.subplots(1, len(names), sharey=True, figsize=(3*len(names), 3))
        
        for i, chain in enumerate(chains):
            maxl = chain.samples[chain.loglike.argmax()]
            mean = chain.samples.mean(axis=0)
            percentiles = np.percentile(chain.samples, percentile_list, axis=0)
            
            sc = to_hex(colors[i]) # Solid color
            fc = sc + '99' # Transparent color
            
            for j, n in enumerate(names):
                if n not in chain.names:
                    self.logger.info(f'Parameter {n} not in chain, skipping.')
                    continue # Skip if the parameter is not in the chain
                
                idx = chain.names.index(n) # Get the right index for the parameter to plot !
                err = np.abs([mean[idx] - percentiles[0][idx], percentiles[1][idx] - mean[idx]]).reshape(2, 1) # 2 values, 1 parameter, expected shape by errorbar
                ax[j].errorbar(mean[idx], i, xerr=err, fmt='', ecolor=sc, lw=2, capsize=5)
                ax[j].plot(maxl[idx], i, 'o', mec=sc, mfc='white', ms=8)
                ax[j].plot(mean[idx], i, 'o', mec=sc, mfc=fc)
                
        # Markers
        markers = kwargs.get('markers', None)
        if markers is not None:
            for i, n in enumerate(names):
                if n not in markers: continue # Skip if the parameter is not in the markers
                ax[i].axvline(markers[n], color='k', linestyle='--', lw=0.8, alpha=0.5)
        
        # Set labels for first plot
        ax[0].set_yticks(np.arange(len(chain_labels)))
        ax[0].set_yticklabels(chain_labels)
        ax[0].set_xlabel(labels[0])
        
        # Make y ticks invisible for all other plots
        for i, a in enumerate(ax[1:], start=1):
            plt.setp(a.get_yticklines(), visible=False)
            a.set_xlabel(labels[i])
        
        fig.tight_layout()
        
        return fig, ax