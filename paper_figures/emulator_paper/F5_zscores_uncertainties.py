import json
import argparse
import numpy as np
from sunbird.data.data_readers import Abacus
from sunbird.summaries import Bundle
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "vibrant"])

args = argparse.ArgumentParser()
args.add_argument("--loss", type=str, default="learned_gaussian")
args = args.parse_args()

with open("../../data/train_test_split.json") as f:
    train_test_split = json.load(f)["test"]
statistic = "density_split_cross"
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
abacus = Abacus(
    select_filters=select_filters,
    slice_filters=slice_filters,
)
parameters = []
for cosmology in train_test_split:
    parameters.append(abacus.get_all_parameters(cosmology=cosmology))
parameters = np.vstack(parameters)

abacus_corrs = []
abacus_corrs = []
for cosmo in train_test_split:
    corrs = abacus.read_statistic(
        statistic=statistic,
        cosmology=cosmo,
        phase=0,
    )
    abacus_corrs.append(corrs.values)
s = corrs.s.values
abacus_corrs = np.array(abacus_corrs)
abacus_corrs = abacus_corrs.reshape((-1, 4, 2, 36))

emulator = Bundle(
    summaries=[statistic],
    loss=args.loss,
)
emu_pred, emu_pred_error = emulator.get_for_batch_inputs(
    parameters,
    select_filters=select_filters,
    slice_filters=slice_filters,
)
abacus_corrs = abacus_corrs.reshape((6, 100, 4, 2, 36))
emu_pred_error = emu_pred_error.reshape((6, 100, 4, 2, 36))


def get_all_zscores():
    quantile_emu_pred = emu_pred[..., :].reshape(-1)
    quantile_abacus_corrs = abacus_corrs[..., :].reshape(-1)
    pred_uncertainty = emu_pred_error[..., :].reshape(-1)
    return (quantile_emu_pred - quantile_abacus_corrs) / pred_uncertainty


zscore = get_all_zscores()
x = np.linspace(-6, 6, 300)
print(np.std(zscore))
print(zscore.shape)
gauss = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
n, bins, patches = plt.hist(
    zscore,
    bins=200,
    density=True,
    alpha=0.5,
)
plt.text(
    2,
    0.3,
    f"$\sigma$ = {np.std(zscore):.2f}",
    color=patches[0].get_facecolor(),
    fontsize=13,
)
plt.plot(x, gauss, linewidth=1, color="k", label=r"$\mathcal{N}(0,1)$")
plt.xlim(-6, 6)
plt.xlabel(r"$\frac{X^\mathrm{emu}-X^\mathrm{test}}{\sigma^\mathrm{emu}}$")
plt.ylabel("PDF")
plt.legend()
plt.savefig(f"figures/png/zscores.png", dpi=300)
plt.savefig(f"figures/pdf/zscores.pdf", dpi=300)
