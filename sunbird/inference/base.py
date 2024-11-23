from getdist import plots, MCSamples


class BaseSampler:
    def __init__(self):
        pass

    def triangle_plot(self, save_fn=None, thin=1, **kwargs):
        """Plot triangle plot
        """
        import matplotlib.pyplot as plt
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        labels = [self.labels[param].strip('$') for param in names]
        samples, weights = self.get_chain(flat=True, thin=thin)
        samples = MCSamples(samples=samples, weights=weights, names=names, labels=labels)
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, **kwargs)
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        plt.show()

    def trace_plot(self, save_fn=None, thin=1):
        """Parameter trace plot
        """
        import matplotlib.pyplot as plt
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        labels = [self.labels[param] for param in names]
        samples, weights = self.get_chain(flat=True, thin=thin)
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 2*len(names)))
        for i, name in enumerate(names):
            ax[i].plot(samples[:, i])
            ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Iteration')
        plt.tight_layout()
        plt.savefig(save_fn, bbox_inches='tight')
        plt.show()

    def save_chain(self, save_fn, thin=1):
        """Save the chain to a file
        """
        import numpy as np
        chain = self.sampler.get_chain(flat=True)
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        cout = { 'samples': chain,
            'weights': np.ones(chain.shape[0]),
            'param_ranges': self.ranges,
            'param_names': names,
            'param_labels': self.labels,
        }
        np.save(save_fn, cout)
