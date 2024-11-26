from getdist import plots, MCSamples


class BaseSampler:
    def __init__(self):
        pass

    def triangle_plot(self, save_fn=None, thin=1, add_bestfit=False, **kwargs):
        """Plot triangle plot
        """
        import matplotlib.pyplot as plt
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        labels = [self.labels[param].strip('$') for param in names]
        data = self.get_chain(flat=True, thin=thin)
        samples = MCSamples(samples=data['samples'], weights=data['weights'], names=names,
                            loglikes=data['log_likelihood'], labels=labels, ranges=self.ranges)
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, **kwargs)
        maxl = data['samples'][data['log_likelihood'].argmax()]
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
        plt.show()

    def trace_plot(self, save_fn=None, thin=1):
        """Parameter trace plot
        """
        import matplotlib.pyplot as plt
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        labels = [self.labels[param] for param in names]
        data = self.get_chain(flat=True, thin=thin)
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 2*len(names)))
        for i, name in enumerate(names):
            ax[i].plot(data['samples'][:, i])
            ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Iteration')
        plt.tight_layout()
        plt.savefig(save_fn, bbox_inches='tight')
        plt.show()

    def save_chain(self, save_fn, metadata=None):
        """Save the chain to a file
        """
        import numpy as np
        data = self.get_chain(flat=True)
        names = [param for param in self.priors.keys() if param not in self.fixed_parameters]
        cout = {**data,
            'ranges': self.ranges,
            'names': names,
            'labels': self.labels,
        }
        if metadata:
            for key, val in metadata.items():
                cout[key] = val
        np.save(save_fn, cout)
