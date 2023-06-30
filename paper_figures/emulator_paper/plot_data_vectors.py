import matplotlib.pyplot as plt
import numpy as np
import colorsys


from sunbird.data.data_readers import Abacus
from sunbird.summaries import DensitySplitCross, DensitySplitAuto, DensityPDF, TPCF
from sunbird.covariance import CovarianceMatrix

plt.style.use(["science", "no-latex"])
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ["lightseagreen", "mediumorchid", "salmon", "royalblue", "rosybrown"]


def get_errors(statistic, dataset):
    covariance = CovarianceMatrix(
        statistics=[statistic],
        covariance_data_class="AbacusSmall",
        dataset=dataset,
        select_filters={"quintiles": [0, 1, 3, 4], "multipoles": [0, 2]},
        slice_filters={"s": [0.7, 150.0]},
    )
    covariance_data = covariance.get_covariance_data(
        apply_hartlap_correction=True,
        volume_scaling=64.0,
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
    cosmology,
    hod_idx,
    emulators,
):
    statistic = "density_pdf"
    yerr = get_errors(statistic, dataset)
    true_statistic, pred_statistic, pred_error_statistic, stat = get_true_and_emu(
        emulators=emulators,
        statistic=statistic,
        cosmology=cosmology,
        hod_idx=hod_idx,
    )
    print(stat.coordinates)
    delta = stat.coordinates["delta"]
    fig = plt.figure()
    c = plt.plot(delta, pred_statistic, label="Predicted")
    plt.fill_between(
        delta,
        pred_statistic - pred_error_statistic,
        pred_statistic + pred_error_statistic,
        alpha=0.6,
        color=c[0].get_color(),
    )
    plt.errorbar(
        delta,
        true_statistic,
        yerr=yerr,
        label="True",
        linestyle="",
        markersize=1,
        marker="o",
        capsize=1.0,
        color=c[0].get_color(),
        alpha=0.5,
    )

    plt.xlim(-1, 5)
    plt.ylabel(r"$\mathcal{P}(\delta), \, R = 10$ Mpc/h")
    plt.xlabel(r"$\delta$")
    plt.legend()
    return fig


def plot_density_multipole(
    dataset, cosmology, hod_idx, multipole, emulators, corr_type="cross"
):
    statistic = f"density_split_{corr_type}"
    yerr = get_errors(statistic, dataset)
    true_statistic, pred_statistic, pred_error_statistic, stat = get_true_and_emu(
        emulators=emulators,
        statistic=statistic,
        cosmology=cosmology,
        hod_idx=hod_idx,
    )
    pred_statistic = pred_statistic.reshape(true_statistic.shape)
    yerr = yerr.reshape(true_statistic.shape)
    s = stat.coordinates["s"]

    fig = plt.figure()
    for quantile in range(4):
        plt.plot(
            s,
            s**2 * pred_statistic[quantile, multipole],
            label=f"DS{quantile}" if quantile < 2 else f"DS{quantile+1}",
            color=colors[quantile],
            linewidth=0.7,
            alpha=0.7,
        )

        plt.errorbar(
            s,
            s**2 * true_statistic[quantile, multipole],
            yerr=s**2 * yerr[quantile, multipole],
            linestyle="",
            markersize=0.8,
            marker="o",
            capsize=0.8,
            color=colors[quantile],
        )
    if corr_type == "cross":
        if multipole == 0:
            label = r"$\xi^{\rm QG}_0(s)$"
        elif multipole == 1:
            label = r"$\xi^{\rm QG}_2(s)$"
    elif corr_type == "auto":
        if multipole == 0:
            label = r"$\xi^{\rm QQ}_0(s)$"
        elif multipole == 1:
            label = r"$\xi^{\rm QQ}_2(s)$"
    plt.ylabel(label)
    plt.xlabel(r"s $[\mathrm{Mpc}/h]$")
    plt.legend(fontsize=7)
    return fig


if __name__ == "__main__":
    cosmology = 0
    hod_idx = 26
    loss = "mae"
    dataset = "bossprior"
    statistic = "density_pdf"
    emulators = {
        "density_pdf": DensityPDF(loss=loss, dataset=dataset),
        "density_split_cross": DensitySplitCross(loss=loss, dataset=dataset),
        "density_split_auto": DensitySplitAuto(loss=loss, dataset=dataset),
        "tpcf": TPCF(loss=loss, dataset=dataset),
    }
    fig = plot_density_multipole(
        dataset=dataset,
        cosmology=cosmology,
        hod_idx=hod_idx,
        emulators=emulators,
        multipole=0,
    )
    plt.savefig(f"figures/png/data_vectors_cross_m0.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_cross_m0.pdf", bbox_inches="tight")
    fig = plot_density_multipole(
        dataset=dataset,
        cosmology=cosmology,
        hod_idx=hod_idx,
        emulators=emulators,
        multipole=1,
    )
    plt.savefig(f"figures/png/data_vectors_cross_m2.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_cross_m2.pdf", bbox_inches="tight")
    fig = plot_density_multipole(
        dataset=dataset,
        cosmology=cosmology,
        hod_idx=hod_idx,
        emulators=emulators,
        corr_type="auto",
        multipole=0,
    )
    plt.savefig(f"figures/png/data_vectors_auto_m0.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_auto_m0.pdf", bbox_inches="tight")
    fig = plot_density_multipole(
        dataset=dataset,
        cosmology=cosmology,
        hod_idx=hod_idx,
        emulators=emulators,
        multipole=1,
        corr_type="auto",
    )
    plt.savefig(f"figures/png/data_vectors_auto_m2.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_auto_m2.pdf", bbox_inches="tight")

    fig = plot_density_pdf(
        dataset=dataset,
        cosmology=cosmology,
        hod_idx=hod_idx,
        emulators=emulators,
    )
    plt.savefig(f"figures/png/data_vectors_density.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs_density.pdf", bbox_inches="tight")
    """
    fig = plot_density_pdf(
        dataset=dataset,
    )
    plt.savefig(f"figures/png/data_vectors.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"figures/pdf/data_vecotrs.pdf", bbox_inches="tight")
    """
