import numpy as np
import random
import json
import pandas as pd


def read_params(cosmo_idx, dataset):
    if dataset == "combined":
        return np.concatenate(
            (
                read_params(cosmo_idx, "same_hods"),
                read_params(cosmo_idx, "different_hods"),
            )
        )
    params = pd.read_csv(
        f"../data/parameters/{dataset}/AbacusSummit_c{cosmo_idx}_hod1000.csv"
    )
    return params.to_numpy()


def read_tpcf_multipoles(
    cosmo_idx,
    dataset,
    multipole=0,
):
    if dataset == "combined":
        s, same = read_multipoles(cosmo_idx, dataset="same_hods", multipole=multipole)
        s, different = read_multipoles(
            cosmo_idx, dataset="different_hods", multipole=multipole
        )
        return s, np.concatenate((same, different))

    data = np.load(
        f"../data/clustering/{dataset}/xi_smu/xi_smu_c{cosmo_idx}_ph000.npy",
        allow_pickle=True,
    ).item()
    return data["s"], np.mean(data["multipoles"], axis=1)[:, multipole]


def read_multipoles(
    cosmo_idx,
    dataset,
    quintile=0,
    multipole=0,
    corr_type="cross",
    filter_type="gaussian",
):
    if dataset == "combined":
        s, same = read_multipoles(
            cosmo_idx, dataset="same_hods", quintile=quintile, multipole=multipole
        )
        s, different = read_multipoles(
            cosmo_idx, dataset="different_hods", quintile=quintile, multipole=multipole
        )
        return s, np.concatenate((same, different))
    else:
        data = np.load(
            f"../data/clustering/{dataset}/ds/{filter_type}/ds_{corr_type}_xi_smu_zsplit_Rs20_c{cosmo_idx}_ph000.npy",
            allow_pickle=True,
        ).item()
        return data["s"], np.mean(data["multipoles"], axis=1)[:, quintile, multipole]


def store_summaries(s, parameters, multipoles, path_to_store):
    summary = {
        "x_min": [float(v) for v in np.min(parameters, axis=0)],
        "x_max": [float(v) for v in np.max(parameters, axis=0)],
        "y_min": float(np.min(multipoles)),
        "s2_y_min": float(np.min(s**2 * multipoles)),
        "s2_y_max": float(np.max(s**2 * multipoles)),
        "y_max": float(np.max(multipoles)),
        "y_mean": [float(v) for v in np.mean(multipoles, axis=0)],
        "y_std": [float(v) for v in np.std(multipoles, axis=0)],
    }
    with open(path_to_store, "w") as fd:
        json.dump(summary, fd)


def read_params_for_stage(idx_list, dataset):
    params = []
    for idx in idx_list:
        params.append(read_params(idx, dataset))
    params = np.array(params)
    return params.reshape(-1, params.shape[-1])


def read_tpcf_for_stage(
    idx_list,
    dataset,
):
    multipoles = []
    for idx in idx_list:
        combined_multi = []
        for multipole in [0, 1]:
            s, multi = read_tpcf_multipoles(idx, multipole=multipole, dataset=dataset)
            combined_multi.append(multi)
        combined_multi = np.hstack(combined_multi)
        multipoles.append(combined_multi)
    multipoles = np.array(multipoles)
    s = np.array(list(s) + list(s))
    multipoles = multipoles.reshape((-1, multipoles.shape[-1]))
    return s, multipoles


def read_multipoles_for_stage(idx_list, dataset, quintile, corr_type, filter_type):
    multipoles = []
    for idx in idx_list:
        combined_multi = []
        for multipole in [0, 1]:
            s, multi = read_multipoles(
                idx,
                multipole=multipole,
                quintile=quintile,
                dataset=dataset,
                corr_type=corr_type,
                filter_type=filter_type,
            )
            combined_multi.append(multi)
        combined_multi = np.hstack(combined_multi)
        multipoles.append(combined_multi)
    multipoles = np.array(multipoles)
    s = np.array(list(s) + list(s))
    multipoles = multipoles.reshape((-1, multipoles.shape[-1]))
    return s, multipoles


if __name__ == "__main__":
    quintiles = [0, 1, 3, 4]
    filter_type = "gaussian"
    corr_types = ["auto", "cross"]
    dataset = "different_hods_linsigma"
    with open("../data/train_test_split.json") as f:
        train_test_split = json.load(f)
    train_params = read_params_for_stage(train_test_split["train"], dataset=dataset)
    np.save(
        f"../data/datasets/{dataset}/train_params.npy",
        train_params,
    )
    for stage in ["test", "val"]:
        params = read_params_for_stage(train_test_split[stage], dataset=dataset)
        np.save(
            f"../data/datasets/{dataset}/{stage}_params.npy",
            params,
        )
    s, train_multipoles = read_tpcf_for_stage(
        train_test_split["train"], dataset=dataset
    )
    np.save(f"../data/datasets/{dataset}/train_tpcf.npy", train_multipoles)
    np.save(f"../data/s.npy", np.unique(s))
    store_summaries(
        s,
        train_params,
        train_multipoles,
        f"../data/datasets/{dataset}/train_tpcf_summary.json",
    )

    for stage in ["test", "val"]:
        s, multipoles = read_tpcf_for_stage(train_test_split[stage], dataset=dataset)
        np.save(f"../data/datasets/{dataset}/{stage}_tpcf.npy", multipoles)

    for corr_type in corr_types:
        for quintile in quintiles:
            s, train_multipoles_quintile = read_multipoles_for_stage(
                train_test_split["train"],
                dataset=dataset,
                quintile=quintile,
                corr_type=corr_type,
                filter_type=filter_type,
            )
            store_summaries(
                s,
                train_params,
                train_multipoles_quintile,
                f"../data/datasets/{dataset}/train_ds{quintile}_{corr_type}_summary.json",
            )
            np.save(
                f"../data/datasets/{dataset}/train_ds{quintile}_{corr_type}.npy",
                train_multipoles_quintile,
            )

            for stage in ["test", "val"]:
                s, multipoles = read_multipoles_for_stage(
                    train_test_split[stage],
                    dataset=dataset,
                    quintile=quintile,
                    corr_type=corr_type,
                    filter_type=filter_type,
                )
                np.save(
                    f"../data/datasets/{dataset}/{stage}_ds{quintile}_{corr_type}.npy",
                    multipoles,
                )
