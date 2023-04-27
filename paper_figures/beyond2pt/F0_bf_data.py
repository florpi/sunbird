import matplotlib.pyplot as plt
import numpy as np
from sunbird.covariance import CovarianceMatrix
from sunbird.data.data_readers import Beyond2pt
from sunbird.summaries import Bundle
from F2_data_inference import get_chain_from_full_path

plt.style.use(['science'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

statistic = 'density_split_cross'

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_data_errors(dataset, statistics, select_filters, slice_filters, volume_scaling,):
    covariance = CovarianceMatrix(
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
        covariance_data_class='AbacusSmall',
        dataset=dataset,
    )
    covariance_data = covariance.get_covariance_data(
        apply_hartlap_correction=False,
        volume_scaling=volume_scaling,
    )

    covariance_data += covariance.get_covariance_emulator(
                covariance_data=covariance_data,
                clip_errors=False,
            )
    covariance_data += covariance.get_covariance_simulation(
                apply_hartlap_correction=False,
            )
    return np.sqrt(np.diag(covariance_data))

def get_chain():
    return get_chain_from_full_path(
        '../../scripts/inference/chains/beyond2pt_smin0.70_smax150.00_q0134_m02_density_split_cross_density_split_auto/results.csv'
    )

def get_emulator_prediction(statistic, select_filters, slice_filters):
    chain = get_chain()
    param_names = [name.name for name in chain.paramNames.names]
    parameters = {'nrun': 0., 'N_ur': 2.0328, 'w0_fld': -1., 'wa_fld': 0.}
    posterior_mean = np.mean(chain.samples,axis=0)
    for i, name in enumerate(param_names):
        parameters[name] = posterior_mean[i]
    # Make sure fixed parameters used
    return Bundle(
        summaries=[statistic],
        dataset='beyond2pt'
    )(
        param_dict=parameters,
        select_filters=select_filters,
        slice_filters=slice_filters,
        return_xarray=True,
    )[0]

def get_posterior_samples(statistic, select_filters, slice_filters, n_samples=100,):
    emulator = Bundle(summaries=[statistic], dataset='beyond2pt')
    chain = get_chain()
    param_names = [name.name for name in chain.paramNames.names]
    int_idx = np.random.choice(range(chain.samples.shape[0]), size=n_samples, replace=False,)
    emu_preds = []
    for idx in int_idx:
        parameters = {'nrun': 0., 'N_ur': 2.0328, 'w0_fld': -1., 'wa_fld': 0.}
        for i, name in enumerate(param_names):
            parameters[name] = chain.samples[idx][i]
        # Make sure fixed parameters used
        emu_preds.append(
            emulator(
                param_dict=parameters,
                select_filters=select_filters,
                slice_filters=slice_filters,
                return_xarray=False,
            )
        )
    return np.array(emu_preds)

if __name__ == '__main__':
    select_filters={'multipoles': [0,2], 'quintiles': [0,1,3,4]}
    slice_filters = {'s': [0.7, 150.]}
    statistic='density_split_cross'
    beyond2pt = Beyond2pt(select_filters=select_filters, slice_filters=slice_filters,)
    data = beyond2pt.read_statistic(
        statistic=statistic,
        multiple_realizations=False,
    )
    data_errors = get_data_errors(
        'beyond2pt',
        statistics=[statistic],
        select_filters=select_filters,
        slice_filters=slice_filters,
        volume_scaling=64.,
    )
    data_errors = data_errors.reshape((4,2,-1))
    s = data.s

    emulator_prediction = get_emulator_prediction(
        statistic=statistic,
        select_filters={'multipoles': [0,2], 'quintiles': [0,1,3,4]},
        slice_filters=slice_filters,
    )
    posterior_samples = get_posterior_samples(
        statistic=statistic,
        select_filters={'multipoles': [0,2], 'quintiles': [0,1,3,4]},
        slice_filters=slice_filters,
    )
    posterior_samples = posterior_samples.reshape((len(posterior_samples), 4,2,-1))
    for iell, ell in enumerate([0, 2]):
        fig, ax = plt.subplots(2, 1, figsize=(4.4,4.), height_ratios=[2.,0.5])
        for ids, ds in enumerate([0,1,3,4]):
            if ds != 2:
                for sample in posterior_samples:
                    ax[0].plot(
                        s,
                        s**2*sample[ids, iell],
                        color=lighten_color(colors[ids], 0.6),
                        alpha=0.1,
                    )
                ax[0].plot(
                    s,
                    s**2* emulator_prediction.sel(
                        quintiles=ds,
                        multipoles=ell,
                    ),
                    color=lighten_color(colors[ids], 1.1),
                    label=rf'${{\rm Q}}_{ds}$',
                )                


                error = (
                    (emulator_prediction.sel(
                        quintiles=ds,
                        multipoles=ell,
                    ) - data.sel(
                        quintiles=ds,
                        multipoles=ell,
                    )) / data_errors[ids, iell]
                )

                ax[1].plot(
                    s,
                    error,
                    color=colors[ids],
                )

            ax[0].plot(
                s, 
                s**2 * data.sel(
                    quintiles=ds,
                    multipoles=ell,
                ),
                ls='none',
                marker='o',
                ms=1.5,
                markeredgewidth=0.2,
                markeredgecolor='gray',
                markerfacecolor='silver',
            )
            ax[0].errorbar(
                s, 
                s**2 * data.sel(
                    quintiles=ds,
                    multipoles=ell,
                ),
                yerr=s**2 * data_errors[ids,iell],
                marker=None,
                ls='none',
                capsize=0.7,
                elinewidth=0.7,
                ecolor='silver',#lighten_color(colors[ids], 1.1),
            )
        ax[1].axhline(0., color='k',)
        ax[1].fill_between(s,-1,1, alpha=0.3, color='gray')
        ax[1].set_ylim(-5,5)

        leg = ax[0].legend(loc='best', ncol=2, fontsize=15,
        handlelength=0.1, columnspacing=1.0)
        ax[0].set_ylabel(rf'$s^2 \xi_{ell}^{{\rm QG}}(s)\, [h^{{-2}}{{\rm Mpc}}^2]$')
        #ax.axes.get_xaxis().set_visible(False)
        ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
        ax[1].set_ylabel(r'$\frac{\xi_\mathrm{Model} - \xi_\mathrm{Data}}{\sigma}$')
        plt.tight_layout()
        #plt.subplots_adjust(hspace=0.0)

        plt.savefig(f"figures/png/Figure0_data_{ell}.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"figures/pdf/Figure0_data_{ell}.pdf", bbox_inches="tight")

