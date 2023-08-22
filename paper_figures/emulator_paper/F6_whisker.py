import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from inference_plot_utils import *

plt.style.use(["science.mplstyle"])


args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/enrique/",
)
args = args.parse_args()

chain_dir = Path(args.chain_dir)

chain_handles = [
    "abacus_cosmo0_hod26_tpcf_learned_gaussian_vol64_smin0.70_smax150.00_m02_q0134",
    "abacus_cosmo0_hod26_density_split_cross_mae_vol64_smin0.70_smax150.00_m02_q0134",
    "abacus_cosmo0_hod26_density_split_cross_tpcf_mae_vol64_smin0.70_smax150.00_m02_q0134",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134_nosimerr",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134_noemuerr",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q04",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q4",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin30.00_smax150.00_m02_q0134",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax30.00_m02_q0134",
    "abacus_cosmo0_hod26_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134",
]

chain_labels = [
    "Galaxy 2PCF only",
    "DS CCF only",
    "DS CCF + Galaxy 2PCF",
    "No simulation error",
    "No emulator error",
    r"${\rm Q_0 + Q_4}$" " only",
    r"${\rm Q_0}$" " only",
    r"${\rm Q_4}$" " only",
    r"$s_{\rm max}=30\,h^{-1}{\rm Mpc}$",
    r"$s_{\rm min}=30\,h^{-1}{\rm Mpc}$",
    "Baseline",
]

redshift = 0.5

yvals = np.linspace(0, 10, len(chain_handles))
params_toplot = ["omega_cdm", "sigma8_m", "n_s", "fsigma8"]
labels_toplot = [r"$\omega_{\rm cdm}$", r"$\sigma_8$", r"$n_s$", r"$f\sigma_8$"]

true_params = get_true_params(
    cosmology=0, hod_idx=26, add_fsigma8=True, redshift=redshift
)

fig, ax = plt.subplots(1, len(params_toplot), figsize=(2.5 * len(params_toplot), 4.5))
for iparam, param in enumerate(params_toplot):
    for ichain, chain_handle in enumerate(chain_handles):
        chain_fn = chain_dir / chain_handle / "results.csv"
        # names, chain, weights = read_dynesty_chain(chain_fn, add_fsigma8=True if param == 'fsigma8' else False)
        samples = get_MCSamples(
            chain_fn,
            add_fsigma8=True,
            redshift=redshift,
        )

        if ichain == len(chain_handles) - 1:
            ax[iparam].fill_betweenx(
                yvals,
                samples.mean(param) - samples.std(param),
                samples.mean(param) + samples.std(param),
                alpha=0.2,
                color="gray",
                edgecolor=None,
            )
            ax[iparam].plot(
                [true_params[param]] * len(yvals),
                yvals,
                color="gray",
                linestyle="dashed",
                alpha=0.3,
            )
        ax[iparam].errorbar(
            samples.mean(param),
            yvals[ichain],
            xerr=samples.std(param),
            marker="o",
            ms=5.0,
            color="indigo",
            capsize=3,
        )

    ax[iparam].set_xlabel(labels_toplot[iparam], fontsize=20)
    ax[iparam].tick_params(axis="x", labelsize=13, rotation=45)
    if iparam > 0:
        ax[iparam].axes.get_yaxis().set_visible(False)
    else:
        ax[iparam].set_yticks(yvals)
        ax[iparam].set_yticklabels(chain_labels, minor=False, rotation=0, fontsize=13)

plt.tight_layout()
plt.subplots_adjust(wspace=0.125)
plt.savefig("figures/pdf/whisker.pdf", bbox_inches="tight")
plt.savefig(f"figures/png/whisker.png", bbox_inches="tight", dpi=300)
