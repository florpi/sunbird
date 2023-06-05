import matplotlib.pyplot as plt
import numpy as np
import colorsys


from sunbird.data.data_readers import Abacus
from sunbird.summaries import DensitySplitCross, DensitySplitAuto, DensityPDF, TPCF
from sunbird.covariance import CovarianceMatrix 
plt.style.use(["science"])

def get_errors(statistic, dataset):
    covariance = CovarianceMatrix(
        statistics=[statistic],
        covariance_data_class='AbacusSmall',
        dataset=dataset,
        select_filters={'quintiles': [0,1,3,4], 'multipoles': [0,2]},
        slice_filters={'s': [0.7,150.]},
    )
    covariance_data = covariance.get_covariance_data(
        apply_hartlap_correction=True,
        volume_scaling=64.,
    )
    return np.sqrt(np.diag(covariance_data))

def get_true_and_emu(emulators, statistic, cosmology, hod_idx):
    abacus = Abacus(
        dataset=dataset,
    )
    parameters = abacus.get_all_parameters(cosmology=cosmology).iloc[hod_idx]
    true_statistic = abacus.read_statistic(
        statistic,
        cosmology=cosmology,
        phase=0,
    )[hod_idx]
    stat = emulators[statistic]
    pred_statistic, pred_error_statistic = stat(parameters)
    return true_statistic, pred_statistic.numpy(), pred_error_statistic.numpy(), stat

def plot_density_pdf(
    dataset,
    cosmology_low,
    hod_idx_low,
    cosmology_high,
    hod_idx_high,
    emulators,
):
    statistic = 'density_pdf'
    yerr = get_errors(statistic, dataset)
    true_statistic_low, pred_statistic_low, pred_error_statistic_low, stat = get_true_and_emu(
        emulators=emulators,
        statistic=statistic,
        cosmology=cosmology_low,
        hod_idx = hod_idx_low,
    )
    true_statistic_high, pred_statistic_high, pred_error_statistic_high, stat = get_true_and_emu(
        emulators=emulators,
        statistic=statistic,
        cosmology=cosmology_high,
        hod_idx = hod_idx_high,
    )

    delta = stat.coordinates['delta']
    fig = plt.figure()
    c = plt.plot(
        delta,
        pred_statistic_low,
        label='Predicted'
    )
    plt.fill_between(
        delta,
        pred_statistic_low - pred_error_statistic_low,
        pred_statistic_low + pred_error_statistic_low,
        alpha=0.6,
        color=c[0].get_color(),
    )
    plt.errorbar(
        delta,
        true_statistic_low,
        yerr=yerr,
        label='True',
        linestyle='',
        markersize=1,
        marker='o',
        capsize=1.,
        color=c[0].get_color(),
        alpha=0.5,
    )
    c = plt.plot(
        delta,
        pred_statistic_high,
    )
    plt.fill_between(
        delta,
        pred_statistic_high - pred_error_statistic_high,
        pred_statistic_high + pred_error_statistic_high,
        alpha=0.6,
        color=c[0].get_color(),
    )
    plt.errorbar(
        delta,
        true_statistic_high,
        yerr=yerr,
        linestyle='',
        markersize=1,
        marker='o',
        capsize=1.,
        color=c[0].get_color(),
        alpha=0.5,
    )
    #plt.yscale('log')
    plt.xlim(-1,5)
    plt.ylabel(r'$\mathcal{P}(\delta), \, R = 10$ Mpc/h')
    plt.xlabel(r'$\delta$')
    plt.legend()

    return fig



def plot_density_cross(
    dataset,
    cosmology_low,
    hod_idx_low,
    cosmology_high,
    hod_idx_high,
    emulators,
):
    statistic = 'density_split_cross'
    yerr = get_errors(statistic, dataset)
    print(yerr.shape)
    true_statistic_low, pred_statistic_low, pred_error_statistic_low, stat = get_true_and_emu(
        emulators=emulators,
        statistic=statistic,
        cosmology=cosmology_low,
        hod_idx = hod_idx_low,
    )
    pred_statistic_low = pred_statistic_low.reshape(true_statistic_low.shape)
    pred_error_statistic_low = pred_error_statistic_low.reshape(true_statistic_low.shape)
    yerr = yerr.reshape(true_statistic_low.shape)
    print('true')
    print(true_statistic_low.shape)
    print('pred')
    print(pred_statistic_low.shape)
    print('pred err')
    print(pred_error_statistic_low.shape)
    true_statistic_high, pred_statistic_high, pred_error_statistic_high, stat = get_true_and_emu(
        emulators=emulators,
        statistic=statistic,
        cosmology=cosmology_high,
        hod_idx = hod_idx_high,
    )
    pred_statistic_high = pred_statistic_high.reshape(true_statistic_low.shape)
    pred_error_statistic_high = pred_error_statistic_high.reshape(true_statistic_low.shape)

    s = stat.coordinates['s']
    multipole = 1
    lightness_values = [0.6, 0.7, 0.8, 0.9]
    base_hue = 0.67
    rgb_values = [colorsys.hls_to_rgb(base_hue, l, 1) for l in lightness_values]

    base_hue_high = 0.33
    rgb_values_high = [colorsys.hls_to_rgb(base_hue_high, l, 1) for l in lightness_values]

    fig = plt.figure()
    for quantile in range(4):
        c = plt.fill_between(
            s,
            #s**2*(pred_statistic_low[quantile,multipole] - pred_error_statistic_low[quantile,multipole]),
            (pred_statistic_low[quantile,multipole] - pred_error_statistic_low[quantile,multipole]),
            #s**2*(pred_statistic_low[quantile,multipole] + pred_error_statistic_low[quantile,multipole]),
            (pred_statistic_low[quantile,multipole] + pred_error_statistic_low[quantile,multipole]),
            alpha=0.5,
            color=rgb_values[quantile],
        )
        plt.plot(
            s,
            pred_statistic_low[quantile, multipole],
            #s**2*pred_statistic_low[quantile, multipole],
            label='Predicted' if quantile == 0 else None,
            color=rgb_values[quantile],
            linewidth=0.7,
            alpha=0.7,
        )

        plt.errorbar(
            s,
            #s**2*true_statistic_low[quantile,multipole],
            true_statistic_low[quantile,multipole],
            #yerr=s**2*yerr[quantile, multipole],
            yerr=yerr[quantile, multipole],
            label='True' if quantile == 0 else None,
            linestyle='',
            markersize=0.8,
            marker='o',
            capsize=0.8,
            color=rgb_values[quantile],
            #alpha=0.5,
        )        
        '''
        c = plt.fill_between(
            s,
            pred_statistic_high[quantile, multipole] - pred_error_statistic_high[quantile, multipole],
            pred_statistic_high[quantile, multipole] + pred_error_statistic_high[quantile, multipole],
            alpha=0.5,
            color=rgb_values_high[quantile]
        )
        plt.plot(
            s,
            pred_statistic_high[quantile, multipole],
            color=rgb_values_high[quantile],
        )

        plt.errorbar(
            s,
            true_statistic_high[quantile,multipole],
            yerr=yerr[quantile, multipole],
            linestyle='',
            markersize=1,
            marker='o',
            capsize=1.,
            color=rgb_values_high[quantile],
            #alpha=0.5,
        )        
        '''
    plt.xlabel('$r$ [Mpc/h]')
    plt.xlim(0,50)
    plt.legend(fontsize=7)
    return fig





if __name__ == '__main__':
    cosmology_low = 3
    cosmology_high = 0
    hod_idx_low = 0  
    hod_idx_high = 26
    loss = 'learned_gaussian'
    dataset = 'bossprior'
    statistic = 'density_pdf'
    emulators = {
        'density_pdf': DensityPDF(loss=loss,dataset=dataset),
        'density_split_cross': DensitySplitCross(loss=loss,dataset=dataset),
        'density_split_auto': DensitySplitAuto(loss=loss,dataset=dataset),
        'tpcf': TPCF(loss=loss,dataset=dataset),
    }
    fig = plot_density_cross(
        dataset=dataset,
        cosmology_low=cosmology_low,
        hod_idx_low=hod_idx_low,
        cosmology_high = cosmology_high,
        hod_idx_high= hod_idx_high,
        emulators=emulators,
    )
    plt.savefig(f"figures/png/data_vectors_cross.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_cross.pdf", bbox_inches="tight")

    fig = plot_density_pdf(
        dataset=dataset,
        cosmology_low=cosmology_low,
        hod_idx_low=hod_idx_low,
        cosmology_high = cosmology_high,
        hod_idx_high= hod_idx_high,
        emulators=emulators,
    )
    plt.savefig(f"figures/png/data_vectors_density.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_density.pdf", bbox_inches="tight")
    '''
    fig = plot_density_pdf(
        dataset=dataset,
    )
    plt.savefig(f"figures/png/data_vectors.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs.pdf", bbox_inches="tight")
    '''
