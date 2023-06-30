import json
import numpy as np
from sunbird.data.data_readers import Abacus
from sunbird.summaries import Bundle
from sunbird.covariance import CovarianceMatrix
import matplotlib.pyplot as plt

plt.style.use(
    [
        "science.mplstyle",
    ]
)  


def get_data_errors(
    dataset,
    statistics,
    select_filters,
    slice_filters,
    volume_scaling,
):
    covariance = CovarianceMatrix(
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
        covariance_data_class="AbacusSmall",
        dataset=dataset,
    )
    covariance_data = covariance.get_covariance_data(
        apply_hartlap_correction=True,
        volume_scaling=volume_scaling,
    )
    return np.sqrt(np.diag(covariance_data))


def get_abacus_parameters(dataset, split="test"):
    with open("../../data/train_test_split.json") as f:
        train_test_split = json.load(f)
    abacus = Abacus(
        dataset=dataset,
    )
    parameters = []
    for cosmology in train_test_split[split]:
        parameters.append(abacus.get_all_parameters(cosmology=cosmology))
    return np.vstack(parameters)


def get_abacus_corrs(dataset, statistics, select_filters, slice_filters, split="test"):
    with open("../../data/train_test_split.json") as f:
        train_test_split = json.load(f)
    abacus = Abacus(
        dataset=dataset,
        select_filters=select_filters,
        slice_filters=slice_filters,
    )
    abacus_corrs = []
    for statistic in statistics:
        corrs_for_stat = []
        for cosmo in train_test_split[split]:
            corrs = abacus.read_statistic(
                statistic=statistic,
                cosmology=cosmo,
                phase=0,
            )
            corrs_for_stat.append(corrs.values)
        abacus_corrs.append(corrs_for_stat)
    s = corrs.s.values
    abacus_corrs = np.array(abacus_corrs)
    return s, abacus_corrs.swapaxes(0, 1).swapaxes(1, 2)


def get_emulator_predictions(
    statistics,
    parameters,
    select_filters,
    slice_filters,
    dataset,
    loss="mae",
):
    emulator = Bundle(
        summaries=statistics,
        dataset=dataset,
        loss=loss,
    )
    return emulator.get_for_batch_inputs(
        parameters,
        select_filters=select_filters,
        slice_filters=slice_filters,
    )


def get_emulator_errors(emulated_corrs, emulated_error_pred, abacus_corrs):
    emulated_corrs = emulated_corrs.reshape(abacus_corrs.shape)
    emulated_error_pred = emulated_error_pred.reshape(abacus_corrs.shape)

    frac_abs_error = np.abs(emulated_corrs - abacus_corrs) / np.abs(abacus_corrs)
    frac_abs_error = frac_abs_error.reshape((-1, 2, 4, 2, 36))
    std_abs_error = np.abs(emulated_corrs - abacus_corrs)
    std_abs_error = std_abs_error.reshape((-1, 2, 4, 2, 36)) / data_std.reshape(
        (1, 2, 4, 2, 36)
    )

    median_frac_error = np.median(frac_abs_error, axis=0)
    median_std_error = np.median(std_abs_error, axis=0)

    frac_error_pred = emulated_error_pred / np.abs(abacus_corrs)
    frac_error_pred = frac_error_pred.reshape((-1, 2, 4, 2, 36))
    median_frac_error_pred = np.median(frac_error_pred, axis=0)

    std_error_pred = emulated_error_pred.reshape((-1, 2, 4, 2, 36)) / data_std.reshape(
        (1, 2, 4, 2, 36)
    )
    median_std_error_pred = np.median(std_error_pred, axis=0)
    return (
        median_frac_error,
        median_std_error,
        median_frac_error_pred,
        median_std_error_pred,
    )


if __name__ == "__main__":
    dataset = "bossprior"
    loss = "learned_gaussian"
    plot_predicted_errors = False
    suffix = None
    volume_scaling = 64.0  # 8. for BOSS volume, 64 for AbacusSummit/Beyond2pt volume
    select_filters = {
        "quintiles": [
            0,
            1,
            3,
            4,
        ],
        "multipoles": [
            0,
            2,
        ],
    }
    slice_filters = {
        "s": [0.7, 150.0],
    }
    statistics = ["density_split_auto", "density_split_cross"]
    data_std = get_data_errors(
        dataset=dataset,
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
        volume_scaling=volume_scaling,
    )
    parameters = get_abacus_parameters(dataset=dataset, split="test")
    s, abacus_corrs = get_abacus_corrs(
        dataset=dataset,
        statistics=statistics,
        select_filters=select_filters,
        slice_filters=slice_filters,
        split="test",
    )
    emulated_corrs, emulated_error_pred = get_emulator_predictions(
        statistics=statistics,
        parameters=parameters,
        select_filters=select_filters,
        slice_filters=slice_filters,
        dataset=dataset,
        loss=loss,
    )
    (
        median_frac_error,
        median_std_error,
        median_frac_error_pred,
        median_std_error_pred,
    ) = get_emulator_errors(
        emulated_corrs=emulated_corrs,
        emulated_error_pred=emulated_error_pred,
        abacus_corrs=abacus_corrs,
    )
    fig, ax = plt.subplots(nrows=4, figsize=(9, 6), sharex=True, sharey=True)
    ds_colors = ["lightseagreen", "mediumorchid", "salmon", "royalblue", "rosybrown"]

    x_range = np.arange(len(median_frac_error[:, :, 0, :].reshape(-1)))
    for i in range(4):
        ax[i].fill_between(
            x_range[: len(x_range) // 2],
            0,
            0.05,
            color="gray",
            alpha=0.1,
            edgecolor=None,
        )
        ax[i].fill_between(
            x_range[: len(x_range) // 2],
            0,
            0.01,
            color="gray",
            alpha=0.2,
            edgecolor=None,
        )

    lw = 1.5

    quintiles = [1, 2, 4, 5]
    for q in range(4):
        ax[0].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_frac_error[0, q, 0, :],
            color=ds_colors[q],
            label=rf"$\mathrm{{DS}}{quintiles[q]}$",
            linewidth=lw,
        )
        ax[1].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_frac_error[0, q, 1, :],
            color=ds_colors[q],
            linewidth=lw,
        )
        ax[2].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_frac_error[1, q, 0, :],
            color=ds_colors[q],
            linewidth=lw,
        )
        ax[3].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_frac_error[1, q, 1, :],
            color=ds_colors[q],
            linewidth=lw,
        )
        if plot_predicted_errors:
            ax[0].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_frac_error_pred[0, q, 0, :],
                color=ds_colors[q],
                linestyle="dashed",
            )
            ax[1].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_frac_error_pred[0, q, 1, :],
                color=ds_colors[q],
                linestyle="dashed",
            )
            ax[2].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_frac_error_pred[1, q, 0, :],
                color=ds_colors[q],
                linestyle="dashed",
            )
            ax[3].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_frac_error_pred[1, q, 1, :],
                color=ds_colors[q],
                linestyle="dashed",
            )

    ax[0].set_ylabel(r"$|\Delta \xi^\mathrm{QQ}_0| / \xi^\mathrm{QQ}_0$")
    ax[1].set_ylabel(r"$|\Delta \xi^\mathrm{QQ}_2| / \xi^\mathrm{QQ}_2$")
    ax[2].set_ylabel(r"$|\Delta \xi^\mathrm{QG}_0| / \xi^\mathrm{QG}_0$")
    ax[3].set_ylabel(r"$|\Delta \xi^\mathrm{QG}_2| / \xi^\mathrm{QG}_2$")
    ax[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.5))
    current_labels = [13, 13, 11] * 4
    current_labels = np.cumsum(current_labels)
    current_labels = [0] + list(current_labels)
    all_s = np.array(list(s) * 4)
    _ = ax[-1].set_xticks(
        current_labels[:-1], [all_s[int(c)] - 0.5 for c in current_labels[:-1]]
    )

    ax[-1].set_xlabel(r"s $[\mathrm{Mpc}/h]$")

    for i in range(4):
        ax[i].set_ylim(0.0, 0.2)
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.tight_layout()

    plt.savefig(
        f"figures/png/F1_emulator_errors_frac.png", dpi=600, bbox_inches="tight"
    )
    plt.savefig(f"figures/pdf/F1_emulator_errors_frac.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(nrows=4, figsize=(9, 6), sharex=True, sharey=False)

    for i in range(4):
        ax[i].fill_between(
            x_range[: len(x_range) // 2],
            0,
            1,
            color="gray",
            alpha=0.2,
            edgecolor=None,
        )
        ax[i].fill_between(
            x_range[: len(x_range) // 2],
            0,
            2,
            color="gray",
            alpha=0.1,
            edgecolor=None,
        )

    for q in range(4):
        ax[0].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_std_error[0, q, 0, :],
            color=ds_colors[q],
            label=rf"$\mathrm{{DS}}{quintiles[q]}$",
            linewidth=lw,
        )
        ax[1].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_std_error[0, q, 1, :],
            color=ds_colors[q],
            linewidth=lw,
        )
        ax[2].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_std_error[1, q, 0, :],
            color=ds_colors[q],
            label=rf"$\mathrm{{DS}}{quintiles[q]}$",
            linewidth=lw,
        )
        ax[3].plot(
            x_range[q * len(s) : (q + 1) * len(s)],
            median_std_error[1, q, 1, :],
            color=ds_colors[q],
            linewidth=lw,
        )
        if plot_predicted_errors:
            ax[0].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_std_error_pred[0, q, 0, :],
                color=ds_colors[q],
                linestyle="dashed",
            )
            ax[1].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_std_error_pred[0, q, 1, :],
                color=ds_colors[q],
                linestyle="dashed",
            )
            ax[2].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_std_error_pred[1, q, 0, :],
                color=ds_colors[q],
                linestyle="dashed",
            )
            ax[3].plot(
                x_range[q * len(s) : (q + 1) * len(s)],
                median_std_error_pred[1, q, 1, :],
                color=ds_colors[q],
                linestyle="dashed",
            )

    ax[0].set_ylabel(r"$|\Delta \xi^\mathrm{QQ}_0| / \sigma_\mathrm{data}$")
    ax[1].set_ylabel(r"$|\Delta \xi^\mathrm{QQ}_2| / \sigma_\mathrm{data}$")
    ax[2].set_ylabel(r"$|\Delta \xi^\mathrm{QG}_0| / \sigma_\mathrm{data}$")
    ax[3].set_ylabel(r"$|\Delta \xi^\mathrm{QG}_2| / \sigma_\mathrm{data}$")

    ax[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.5))
    fig.canvas.draw()
    current_labels = [13, 13, 11] * 4
    current_labels = np.cumsum(current_labels)
    current_labels = [0] + list(current_labels)
    all_s = np.array(list(s) * 4)
    _ = ax[-1].set_xticks(
        current_labels[:-1], [all_s[int(c)] - 0.5 for c in current_labels[:-1]]
    )

    ax[-1].set_xlabel(r"s $[\mathrm{Mpc}/h]$")

    for i in range(4):
        ax[i].set_ylim(0, 3.2)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    plt.savefig(
        f"figures/png/F1_emulators_errors_std.png", dpi=600, bbox_inches="tight"
    )
    plt.savefig(f"figures/pdf/F1_emulators_errors_std.pdf", bbox_inches="tight")
