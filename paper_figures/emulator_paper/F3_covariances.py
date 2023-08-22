import numpy as np
import matplotlib.pyplot as plt
from sunbird.covariance import CovarianceMatrix

if __name__ == "__main__":
    statistics = [
        "density_split_cross",
    ]  # 'density_split_auto',]
    errors = []
    for statistic in statistics:
        for quantile in [
            0,
        ]:  # 1,3,4]):
            cov = CovarianceMatrix(
                statistics=[statistic],
                slice_filters={"s": [0.7, 150.0]},
                select_filters={
                    "multipoles": [
                        0,
                    ],
                    "quintiles": [quantile],
                },
            )
            data_cov = cov.get_covariance_data(volume_scaling=64.0)
            emu_cov = cov.get_covariance_emulator()
            sim_cov = cov.get_covariance_simulation()
            s = cov.emulators[statistic].coordinates["s"]

            plt.plot(
                s,
                np.sqrt(np.diag(data_cov)),
                label=r"$\sigma_{\rm data}$" if quantile == 0 else None,
            )
            plt.plot(
                s,
                np.sqrt(np.diag(data_cov) + np.diag(sim_cov)),
                label=r"$\sqrt{\sigma_{\rm data}^2 + \sigma_{\rm sim}}$"
                if quantile == 0
                else None,
            )
            plt.plot(
                s,
                np.sqrt(np.diag(emu_cov)),
                label=r"$\sigma_{\rm emu}$" if quantile == 0 else None,
            )
        plt.xlabel(r"$s$ [Mpc/h]")
        plt.ylabel(r"$\sigma_{Q_0G}$")
        plt.legend()
        plt.savefig("figures/pdf/covariances.pdf", bbox_inches="tight")
        plt.savefig(f"figures/png/covariances.png", bbox_inches="tight", dpi=300)
