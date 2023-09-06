import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from inference_plot_utils import *
from utils import get_names_labels

plt.style.use(["science.mplstyle"])


args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/emulator_paper/",
)
args.add_argument(
    '--loss',
    type=str,
    default='learned_gaussian',
)
args = args.parse_args()

chain_dir = Path(args.chain_dir)

chain_handles = [
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=density_split_cross-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=0-se=0',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=0',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=0-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=0-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=04-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=4-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=50.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=50.0-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=80.0-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
    f'cos=0-h=26-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1',
]

chain_labels = [
    "Galaxy 2PCF only",
    "DS CCF + ACF only",
    "DS ACF only",
    "DS CCF only",
    "No simulation + No emulator error",
    "No simulation error",
    "No emulator error",
    "Monopole only",
    r"${\rm Q_0 + Q_4}$" " only",
    r"${\rm Q_0}$" " only",
    r"${\rm Q_4}$" " only",
    r"$s_{\rm max}=50\,h^{-1}{\rm Mpc}$",
    r"$s_{\rm min}=50\,h^{-1}{\rm Mpc}$",
    r"$s_{\rm min}=80\,h^{-1}{\rm Mpc}$",
    "Baseline",
]

redshift = 0.5

yvals = np.linspace(0, 10, len(chain_handles))
params_toplot = ["omega_cdm", "sigma8_m", "n_s","fsigma8", "N_ur", "B_sat",] 
labels_toplot =  ['$' + label + '$' for label in get_names_labels(params_toplot)]

true_params = get_true_params(
    cosmology=0, hod_idx=26, add_fsigma8=True, redshift=redshift
)

fig, ax = plt.subplots(1, len(params_toplot), figsize=(2.5 * len(params_toplot), 4.5))
for iparam, param in enumerate(params_toplot):
    for ichain, chain_handle in enumerate(chain_handles):
        chain_fn = chain_dir / chain_handle / "results.csv"
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
            #ax[iparam].plot(
            #    [true_params[param]] * len(yvals),
            #    yvals,
            #    color="gray",
            #    linestyle="dashed",
            #    alpha=0.3,
            #)
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
plt.savefig("figures/pdf/F6_whisker.pdf", bbox_inches="tight")
plt.savefig(f"figures/png/F6_whisker.png", bbox_inches="tight", dpi=300)
