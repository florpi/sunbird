import numpy as np
import random
import json
import pandas as pd


def read_params(cosmo_idx, dataset):
    if dataset=='combined':
        return np.concatenate((read_params(cosmo_idx, 'same_hods'), read_params(cosmo_idx, 'different_hods')))
    params = pd.read_csv(
        f"../data/parameters/{dataset}/AbacusSummit_c{cosmo_idx}_hod1000.csv"
    )
    return params.to_numpy()

def read_tpcf_multipoles(
    cosmo_idx,
    multipole=0,
):
    data = np.load(
        f"../data/clustering/xi_smu/xi_smu_c{cosmo_idx}_ph000.npy",
        allow_pickle=True,
    ).item()
    return data['s'], np.mean(data["multipoles"], axis=1)[:, multipole]


def read_multipoles(
    cosmo_idx,
    dataset,
    quintile=0,
    multipole=0,
):
    if dataset == 'combined':
        s, same = read_multipoles(cosmo_idx, dataset='same_hods', quintile=quintile, multipole=multipole) 
        s, different = read_multipoles(cosmo_idx, dataset='different_hods', quintile=quintile, multipole=multipole) 
        return s, np.concatenate((same, different))
    else:
        data = np.load(
            f"../data/clustering/density_split/{dataset}/ds_cross_xi_smu_zsplit_Rs20_c{cosmo_idx}_ph000.npy",
            allow_pickle=True,
        ).item()
        return data['s'], np.mean(data["multipoles"], axis=1)[:, quintile, multipole]



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
    dataset = 'combined'


    np.save(f"../data/datasets/{dataset}/train_params.npy", params[train_idx].reshape(-1, params.shape[-1]))
    np.save(f"../data/datasets/{dataset}/test_params.npy", params[test_idx].reshape(-1, params.shape[-1]))
    np.save(
        f"../data/datasets/{dataset}/val_params.npy",
        params[val_idx].reshape(-1, params.shape[-1]),
    )
    '''
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
    '''

    for quintile in quintiles:
        multipoles = []
        for idx in cosmo_idx:
            combined_multi = []
            for multipole in [0, 1]:
                s, multi = read_multipoles(idx, dataset=dataset,quintile=quintile, multipole=multipole)
                combined_multi.append(multi)
            combined_multi = np.hstack(combined_multi)
            multipoles.append(combined_multi)
        multipoles = np.array(multipoles)
        s = np.array(list(s) + list(s))
        np.save(
            f"../data/datasets/{dataset}/train_ds{quintile}.npy",
            multipoles[train_idx].reshape(-1, multipoles.shape[-1]),
        )
        np.save(
            f"../data/datasets/{dataset}/test_ds{quintile}.npy",
            multipoles[test_idx].reshape(-1, multipoles.shape[-1]),
        )
        np.save(
            f"../data/datasets/{dataset}/val_ds{quintile}.npy",
            multipoles[val_idx].reshape(-1, multipoles.shape[-1]),
        )
        store_summaries(
            s,
            params[train_idx].reshape(-1, params.shape[-1]),
            multipoles[train_idx].reshape(-1, multipoles.shape[-1]),
            f"../data/datasets/{dataset}/train_ds{quintile}_summary.json",
        )
