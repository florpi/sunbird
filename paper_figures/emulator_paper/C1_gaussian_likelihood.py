import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sunbird.data.data_readers import AbacusSmall

plt.style.use(['science', 'vibrant'])


def compute_mean_cov(summaries, volume_factor=1.0):
    mean = np.mean(summaries, axis=0)
    covariance = np.cov(summaries.T)
    covariance *= volume_factor
    inverse_covariance = np.linalg.solve(
        covariance, np.eye(len(covariance), len(covariance))
    )
    return mean, covariance, inverse_covariance


def compute_xi2(vector, mean, inverse_covariance):
    return (vector - mean) @ inverse_covariance @ (vector - mean).T


def sample_from_multigaussian(mean, covariance, n_samples):
    return np.random.multivariate_normal(mean, covariance, size=n_samples)


def plot_gaussianity(summaries, ax, colors=["#4165c0", "#e770a2"]):
    mean, covariance, inverse_covariance = compute_mean_cov(summaries)

    generated_fiducial = sample_from_multigaussian(
        mean,
        covariance,
        n_samples=len(summaries),
    )

    xi2_data = [compute_xi2(fid, mean, inverse_covariance) for fid in summaries]

    xi2_random = [
        compute_xi2(fid, mean, inverse_covariance) for fid in generated_fiducial
    ]
    dof = summaries.shape[1]
    x = np.linspace(np.min(xi2_data), np.max(xi2_data))
    bins = ax.hist(
        xi2_data,
        bins=50,
        density=True,
        label="Data",
        alpha=0.5,
        edgecolor=None,
        #color=colors[0],
    )
    ax.hist(
        xi2_random,
        bins=bins[1],
        density=True,
        alpha=0.3,
        label="Gaussian",
        edgecolor=None,
        #color=colors[1],
    )
    ax.plot(x, stats.chi2.pdf(x, dof), label=r"$\chi^2$", color="gray", linewidth=2)


if __name__ == "__main__":
    abacus = AbacusSmall(
        statistics=[
            "density_split_auto",
        ],
        select_filters={
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        dataset="bossprior",
    )
    auto_ds = abacus.gather_summaries_for_covariance()
    abacus = AbacusSmall(
        statistics=[
            "density_split_cross",
        ],
        select_filters={
            "multipoles": [0, 2],
            "quintiles": [0, 1, 3, 4],
        },
        dataset="bossprior",
    )
    cross_ds = abacus.gather_summaries_for_covariance()

    abacus = AbacusSmall(
        statistics=["tpcf"],
        select_filters={
            "multipoles": [0, 2],
        },
        dataset="bossprior",
    )
    tpcf = abacus.gather_summaries_for_covariance()

    fig, ax = plt.subplots(ncols=3, figsize=(15, 4.5))

    plot_gaussianity(
        tpcf.reshape(len(tpcf), -1),
        ax[0],
    )
    plot_gaussianity(cross_ds.reshape(len(cross_ds), -1), ax[1])
    plot_gaussianity(auto_ds.reshape(len(auto_ds), -1), ax[2])

    ax[0].legend()
    ax[0].set_xlabel(
        r"$\chi^2$",
    )
    ax[1].set_xlabel(
        r"$\chi^2$",
    )
    ax[2].set_xlabel(
        r"$\chi^2$",
    )

    ax[0].set_ylabel(
        r"$\mathrm{PDF}$",
    )
    ax[1].set_ylabel(
        r"$\mathrm{PDF}$",
    )
    ax[2].set_ylabel(
        r"$\mathrm{PDF}$",
    )
    ax[0].set_title(
        r"$\mathrm{2PCF}$",
    )

    ax[1].set_title(
        r"$\mathrm{DS}^\mathrm{QG}$",
    )
    ax[2].set_title(
        r"$\mathrm{DS}^\mathrm{QQ}$",
    )

    ax[0].tick_params(
        axis="both",
        which="major",
    )
    ax[1].tick_params(
        axis="both",
        which="major",
    )

    plt.tight_layout()
    plt.savefig("figures/png/C1_gaussianity_likelihood.png", dpi=300)
    plt.savefig("figures/pdf/C1_gaussianity_likelihood.pdf")
