import numpy as np
import matplotlib.pyplot as plt

from utils import get_emulator_and_truth, get_data_variance

plt.style.use(["science"])

# 1) Get set to plot -> test
split = "test"
s, true_density_split, emulated_density_split, _ = get_emulator_and_truth(split=split)

# 4) Get emulator error
error = (emulated_density_split - true_density_split) / true_density_split
std_error = np.std(error, axis=0)
avg_error = np.mean(error, axis=0)


avg_error = avg_error.reshape((2, 4, 2, len(s)))
std_error = std_error.reshape((2, 4, 2, len(s)))

fig, ax = plt.subplots(nrows=4, figsize=(9, 6), sharex=True, sharey=True)
ds_colors = ["lightseagreen", "mediumorchid", "salmon", "royalblue", "rosybrown"]

x_range = np.arange(len(avg_error[0, :, 0].reshape(-1)))
quintiles = [1, 2, 4, 5]
for q in range(4):
    ax[0].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[0, :, 0].reshape(-1) - std_error[0, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[0, :, 0].reshape(-1) + std_error[0, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
        label=rf"$\mathrm{{DS}}{quintiles[q]}$",
    )
    ax[1].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[0, :, 1].reshape(-1) - std_error[0, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[0, :, 1].reshape(-1) + std_error[0, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
    )
    ax[2].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[1, :, 0].reshape(-1) - std_error[1, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[1, :, 0].reshape(-1) + std_error[1, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
    )
    ax[3].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[1, :, 1].reshape(-1) - std_error[1, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[1, :, 1].reshape(-1) + std_error[1, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
    )

ax[0].set_ylabel(r"$\Delta \xi^\mathrm{QQ}_0 / \xi^\mathrm{QQ}_0$")
ax[1].set_ylabel(r"$\Delta \xi^\mathrm{QQ}_2 / \xi^\mathrm{QQ}_2$")
ax[2].set_ylabel(r"$\Delta \xi^\mathrm{QG}_0 / \xi^\mathrm{QG}_0$")
ax[3].set_ylabel(r"$\Delta \xi^\mathrm{QG}_2 / \xi^\mathrm{QG}_2$")
ax[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.3))

current_labels = [13, 13, 11] * 4
current_labels = np.cumsum(current_labels)
current_labels = [0] + list(current_labels)
all_s = np.array(list(s) * 4)
_ = ax[-1].set_xticks(
    current_labels[:-1], [all_s[int(c)] - 0.5 for c in current_labels[:-1]]
)

ax[-1].set_xlabel(r"s $[\mathrm{Mpc}/h]$")

for i in range(4):
    ax[i].axhline(y=0, color="k", linestyle="dotted", alpha=0.25)
    ax[i].axhline(y=-0.01, color="k", linestyle="dashed", alpha=0.25)
    ax[i].axhline(y=0.01, color="k", linestyle="dashed", alpha=0.25)
    ax[i].set_ylim(-0.15, 0.15)
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(f"figures/png/Figure1_errors_frac.png", dpi=600, bbox_inches="tight")
plt.savefig(f"figures/pdf/Figure1_errors_frac.pdf", bbox_inches="tight")
# 5) Get data error


std_cross = get_data_variance("density_split_cross")
std_auto = get_data_variance("density_split_auto")


# Plot error in units of variance
error = emulated_density_split - true_density_split
error[:, : error.shape[-1] // 2] /= std_auto[None]
error[:, error.shape[-1] // 2 :] /= std_cross[None]

std_error = np.std(error, axis=0)
avg_error = np.mean(error, axis=0)

avg_error = avg_error.reshape((2, 4, 2, len(s)))
std_error = std_error.reshape((2, 4, 2, len(s)))


fig, ax = plt.subplots(nrows=4, figsize=(9, 6), sharex=True, sharey=True)
ds_colors = ["lightseagreen", "mediumorchid", "salmon", "royalblue", "rosybrown"]

x_range = np.arange(len(avg_error[0, :, 0].reshape(-1)))
quintiles = [1, 2, 4, 5]
for q in range(4):
    ax[0].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[0, :, 0].reshape(-1) - std_error[0, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[0, :, 0].reshape(-1) + std_error[0, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
        label=rf"$\mathrm{{DS}}{quintiles[q]}$",
    )
    ax[1].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[0, :, 1].reshape(-1) - std_error[0, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[0, :, 1].reshape(-1) + std_error[0, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
    )
    ax[2].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[1, :, 0].reshape(-1) - std_error[1, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[1, :, 0].reshape(-1) + std_error[1, :, 0].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
    )
    ax[3].fill_between(
        x_range[q * len(s) : (q + 1) * len(s)],
        (avg_error[1, :, 1].reshape(-1) - std_error[1, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        (avg_error[1, :, 1].reshape(-1) + std_error[1, :, 1].reshape(-1))[
            q * len(s) : (q + 1) * len(s)
        ],
        alpha=0.5,
        color=ds_colors[q],
    )

ax[0].set_ylabel(r"$\Delta \xi^\mathrm{QQ}_0 / \sigma_\mathrm{data}$")
ax[1].set_ylabel(r"$\Delta \xi^\mathrm{QQ}_2 / \sigma_\mathrm{data}$")
ax[2].set_ylabel(r"$\Delta \xi^\mathrm{QG}_0 / \sigma_\mathrm{data}$")
ax[3].set_ylabel(r"$\Delta \xi^\mathrm{QG}_2 / \sigma_\mathrm{data}$")
ax[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.3))

ax[-1].set_xlabel(r"s $[\mathrm{Mpc}/h]$")

fig.canvas.draw()

current_labels = [13, 13, 11] * 4
current_labels = np.cumsum(current_labels)
current_labels = [0] + list(current_labels)
all_s = np.array(list(s) * 4)
_ = ax[-1].set_xticks(
    current_labels[:-1], [all_s[int(c)] - 0.5 for c in current_labels[:-1]]
)

for i in range(4):
    ax[i].axhline(y=0, color="k", linestyle="dotted", alpha=0.25)
    ax[i].axhline(y=-1, color="k", linestyle="dashed", alpha=0.25)
    ax[i].axhline(y=1, color="k", linestyle="dashed", alpha=0.25)
    ax[i].set_ylim(-2, 2)
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(f"figures/png/Figure1_errors_std.png", dpi=600, bbox_inches="tight")
plt.savefig(f"figures/pdf/Figure1_errors_std.pdf", bbox_inches="tight")
