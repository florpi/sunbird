import numpy as np
import random
import json
import pandas as pd


def read_tpcf_multipoles(
    cosmo_idx,
    multipole=0,
):
    data = np.load(
        f"../data/full_ap/clustering/xi_smu/xi_smu_c{cosmo_idx}_ph000.npy",
        allow_pickle=True,
    ).item()
    return data['s'], np.mean(data["multipoles"], axis=1)[:, multipole]


def read_multipoles(
    cosmo_idx,
    quintile=0,
    multipole=0,
):
    data = np.load(
        f"../data/full_ap/clustering/ds/ds_cross_xi_smu_zsplit_Rs20_c{cosmo_idx}_ph000.npy",
        allow_pickle=True,
    ).item()
    return data['s'], np.mean(data["multipoles"], axis=1)[:, quintile, multipole]


def read_params(cosmo_idx):
    params = pd.read_csv(
        f"../data/full_ap/cosmologies/AbacusSummit_c{cosmo_idx}_hod1000.csv"
    )
    return params.to_numpy()


def store_summaries(s, parameters, multipoles, path_to_store):
    summary = {
        "x_min": [float(v) for v in np.min(parameters, axis=0)],
        "x_max": [float(v) for v in np.max(parameters, axis=0)],
        "y_min": float(np.min(multipoles)),
        "s2_y_min": float(np.min(s**2*multipoles)),
        "s2_y_max": float(np.max(s**2*multipoles)),
        "y_max": float(np.max(multipoles)),
        "y_mean": [float(v) for v in np.mean(multipoles, axis=0)],
        "y_std": [float(v) for v in np.std(multipoles, axis=0)],
    }
    with open(path_to_store, "w") as fd:
        json.dump(summary, fd)


if __name__ == "__main__":
    quintiles = [0, 1, 3, 4]
    n_derivative_grid = 27
    cosmo_idx = list(np.arange(100, 100 + n_derivative_grid)) + list(
        np.arange(130, 181)
    )
    params = []
    for idx in cosmo_idx:
        params.append(read_params(idx))
    params = np.array(params)
    percent_val = 0.1
    percent_test = 0.1
    n_samples = len(params)
    idx = n_derivative_grid + np.arange(n_samples - n_derivative_grid)
    random.shuffle(idx)
    n_val = int(np.floor(percent_val * n_samples))
    n_test = int(np.floor(percent_test * n_samples))
    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = list(idx[n_val + n_test :]) + list(range(n_derivative_grid))

    np.save("../data/train_params.npy", params[train_idx].reshape(-1, params.shape[-1]))
    np.save("../data/test_params.npy", params[test_idx].reshape(-1, params.shape[-1]))
    np.save(
        "../data/val_params.npy",
        params[val_idx].reshape(-1, params.shape[-1]),
    )
    multipoles = []
    for idx in cosmo_idx:
        combined_multi = []
        for multipole in [0, 1]:
            s, multi = read_tpcf_multipoles(idx, multipole=multipole)
            combined_multi.append(multi)
        combined_multi = np.hstack(combined_multi)
        multipoles.append(combined_multi)
    multipoles = np.array(multipoles)
    s = np.array(list(s) + list(s))
    np.save(
        f"../data/train_tpcf.npy",
        multipoles[train_idx].reshape(-1, multipoles.shape[-1]),
    )
    store_summaries(
        s,
        params[train_idx].reshape(-1, params.shape[-1]),
        multipoles[train_idx].reshape(-1, multipoles.shape[-1]),
        f"../data/train_tpcf_summary.json",
    )
    np.save(
        f"../data/test_tpcf.npy",
        multipoles[test_idx].reshape(-1, multipoles.shape[-1]),
    )
    np.save(
        f"../data/val_tpcf.npy",
        multipoles[val_idx].reshape(-1, multipoles.shape[-1]),
    )

    for quintile in quintiles:
        multipoles = []
        for idx in cosmo_idx:
            combined_multi = []
            for multipole in [0, 1]:
                s, multi = read_multipoles(idx, quintile=quintile, multipole=multipole)
                combined_multi.append(multi)
            combined_multi = np.hstack(combined_multi)
            multipoles.append(combined_multi)
        multipoles = np.array(multipoles)
        s = np.array(list(s) + list(s))
        np.save(
            f"../data/train_ds{quintile}.npy",
            multipoles[train_idx].reshape(-1, multipoles.shape[-1]),
        )
        np.save(
            f"../data/test_ds{quintile}.npy",
            multipoles[test_idx].reshape(-1, multipoles.shape[-1]),
        )
        np.save(
            f"../data/val_ds{quintile}.npy",
            multipoles[val_idx].reshape(-1, multipoles.shape[-1]),
        )
        store_summaries(
            s,
            params[train_idx].reshape(-1, params.shape[-1]),
            multipoles[train_idx].reshape(-1, multipoles.shape[-1]),
            f"../data/train_ds{quintile}_summary.json",
        )
