from getdist import plots, MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots

args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/enrique",
)
args = args.parse_args()

cosmologies = [0, 1, 3, 4]

# best hod for each cosmology
best_hod = {0: 26, 1: 74, 3: 30, 4: 15}

chain_dir = Path(args.chain_dir)
chain_labels = [f"c{cosmo:03}" for cosmo in cosmologies]

params_toplot = [
    "omega_cdm",
    "sigma8_m",
]
true_params = [
    inference_plots.get_true_params(cosmo, best_hod[cosmo]) for cosmo in cosmologies
]

samples_list = []
for cosmo in cosmologies:
    hod = best_hod[cosmo]
    chain_fn = (
        chain_dir
        / f"abacus_cosmo{cosmo}_hod{hod}_density_split_cross_density_split_auto_mae_vol64_smin0.70_smax150.00_m02_q0134"
    )
    samples_list.append(
        inference_plots.get_MCSamples(
            chain_fn / "results.csv",
        )
    )

colors = ["#4165c0", "#e770a2", "#5ac3be", "dimgray",] #"#f79a1e"]
ax = inference_plots.plot_corner(
    samples_list,
    params_toplot,
    chain_labels,
    colors=colors,
    true_params=None,
    markers=true_params,
    markers_colors=colors[::-1],
    inches_per_param=15.5 / 7,
)
plt.savefig("figures/pdf/cosmo_inference_c0_c1_c3_c4.pdf", bbox_inches="tight")
plt.savefig("figures/png/cosmo_inference_c0_c1_c3_c4.png", bbox_inches="tight")
