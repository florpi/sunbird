from getdist import plots, MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots
import matplotlib.colors as mcolors
import colorsys


def darken_color(rgb_color, factor=0.7):
    """
    Darkens the given color by multiplying the value by the given factor.

    Args:
    - rgb_color: RGB color to be darkened.
    - factor: Factor by which to darken the color. Default is 0.7, which darkens the color to 70%.

    Returns:
    - darkened_rgb_color: Darkened RGB color.
    """
    r, g, b = mcolors.to_rgb(rgb_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(min(l * factor, 1), 0)  # Ensure lightness stays in the 0-1 range
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return r, g, b


args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/sunbird/chains/emulator_paper/",
)
args.add_argument(
    "--loss",
    type=str,
    default="learned_gaussian",
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
        / f"cos={cosmo}-h={hod}-o=Abacus-l={args.loss}-smin=0.7-smax=150.0-m=02-q=0134-st=tpcf;density_split_cross;density_split_auto-ab=1-vb=1-ete=1-se=1"
    )
    samples_list.append(
        inference_plots.get_MCSamples(
            chain_fn / "results.csv",
        )
    )

colors = [
    "#4165c0",
    "#e770a2",
    "#5ac3be",
    "dimgray",
]  # "#f79a1e"]
darkened_colors = [
    mcolors.to_hex(
        darken_color(
            color,
        )
    )
    for color in colors
]
ax = inference_plots.plot_corner(
    samples_list,
    params_toplot,
    chain_labels,
    colors=colors,
    true_params=None,
    markers=true_params,
    markers_colors=darkened_colors[::-1],
    inches_per_param=15.5 / 7,
)
plt.savefig("figures/pdf/F7_cosmo_c0_c1_c3_c4.pdf", bbox_inches="tight")
plt.savefig("figures/png/F7_cosmo_c0_c1_c3_c4.png", bbox_inches="tight")
