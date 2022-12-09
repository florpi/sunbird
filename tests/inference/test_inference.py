import numpy as np
import pytest

from sunbird.inference import Nested
from sunbird.summaries import DensitySplitCross


def test_observation():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology=124,
        hod_idx=4,
        statistics=["density_split_cross"],
        select_filters=None,
        slice_filters=None,
    )
    path_direct = "../../data/clustering/different_hods/ds/gaussian/ds_cross_xi_smu_zsplit_Rs20_c124_ph000.npy"
    observation_direct = np.load(path_direct, allow_pickle=True).item()
    s = observation_direct["s"]
    observation_direct = np.mean(observation_direct["multipoles"][4], axis=0)
    np.testing.assert_almost_equal(
        observation_from_inference,
        observation_direct.reshape(-1),
    )


def test_observation_with_filters():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology=124,
        hod_idx=4,
        statistics=["density_split_cross"],
        select_filters={"multipoles": [2], "quintiles": [1, 4]},
        slice_filters={"s": [30.0, 60.0]},
    )
    path_direct = "../../data/clustering/different_hods/ds/gaussian/ds_cross_xi_smu_zsplit_Rs20_c124_ph000.npy"
    observation_direct = np.load(path_direct, allow_pickle=True).item()
    s = observation_direct["s"]
    observation_direct = np.mean(observation_direct["multipoles"][4], axis=0)
    observation_direct = observation_direct[..., (s > 30.0) & (s < 60.0)]
    observation_direct = observation_direct[[1, 4], :, :]
    observation_direct = observation_direct[:, 2, :]
    np.testing.assert_almost_equal(
        observation_from_inference,
        observation_direct.reshape(-1),
    )


def test_observation_combined_summaries():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology=124,
        hod_idx=4,
        statistics=["density_split_auto", "density_split_cross"],
        select_filters=None,
        slice_filters=None,
    )
    observations = []
    for corr in ["auto", "cross"]:
        path_direct = f"../../data/clustering/different_hods/ds/gaussian/ds_{corr}_xi_smu_zsplit_Rs20_c124_ph000.npy"
        observation_direct = np.load(path_direct, allow_pickle=True).item()
        s = observation_direct["s"]
        observations.append(np.mean(observation_direct["multipoles"][4], axis=0))
    observations = np.array(observations)
    np.testing.assert_almost_equal(
        observation_from_inference,
        observations.reshape(-1),
    )


def test_observation_with_filters_combined():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology=124,
        hod_idx=4,
        statistics=["density_split_auto", "density_split_cross"],
        select_filters={"multipoles": [2], "quintiles": [1, 4]},
        slice_filters={"s": [30.0, 60.0]},
    )
    observations = []
    for corr in ["auto", "cross"]:
        path_direct = f"../../data/clustering/different_hods/ds/gaussian/ds_{corr}_xi_smu_zsplit_Rs20_c124_ph000.npy"
        observation_direct = np.load(path_direct, allow_pickle=True).item()
        s = observation_direct["s"]
        observation_direct = np.mean(observation_direct["multipoles"][4], axis=0)
        observation_direct = observation_direct[..., (s > 30.0) & (s < 60.0)]
        observation_direct = observation_direct[[1, 4], :, :]
        observation_direct = observation_direct[:, 2, :]
        observations.append(observation_direct)
    observations = np.array(observations)
    np.testing.assert_almost_equal(
        observation_from_inference,
        observations.reshape(-1),
    )


def test__get_parameters_for_abacus():
    params = Nested.get_parameters_for_abacus(
        cosmology=120,
        hod_idx=1,
    )
    true_params_keys = [
        "omega_b",
        "omega_cdm",
        "sigma8_m",
        "n_s",
        "nrun",
        "N_ur",
        "w0_fld",
        "wa_fld",
        "logM1",
        "logM_cut",
        "alpha",
        "alpha_s",
        "alpha_c",
        "logsigma",
        "kappa",
    ]
    true_params_values = [
        0.02237,
        0.12,
        0.808181,
        0.9619,
        0.0,
        2.0328,
        -1.0,
        0.0,
        13.801977238343973,
        12.918160945631291,
        1.3109677996570976,
        0.8889915020022103,
        0.07765655296582806,
        -2.55477928370309,
        0.6259170522372437,
    ]
    true_params = dict(zip(true_params_keys, true_params_values))
    for k, value in params.items():
        assert pytest.approx(true_params[k]) == value


def test_get_covariance_data():
    covariance = Nested.get_covariance_matrix(
        statistics=["tpcf"],
        select_filters=None,
        slice_filters=None,
        add_emulator_error=False,
        apply_hartlap_correction=True,
    )
    measured_for_cov = np.load(
        "../../data/covariance/xi_smu/xi_smu_landyszalay_randomsX50.npy",
        allow_pickle=True,
    ).item()
    expected_covariance = measured_for_cov["multipoles"]
    expected_covariance = np.cov(expected_covariance.reshape((1000,-1)).T)
    hartlap_correction = (1000-1) / (1000 - len(expected_covariance) - 2)
    np.testing.assert_almost_equal(
        hartlap_correction * expected_covariance,
        covariance,
    )

def test_get_covariance_data_with_filters():
    covariance = Nested.get_covariance_matrix(
        statistics=["tpcf"],
        select_filters={"multipoles": [2],},
        slice_filters={"s": [30.0, 60.0]},
        add_emulator_error=False,
        apply_hartlap_correction=True,
    )
    measured_for_cov = np.load(
        "../../data/covariance/xi_smu/xi_smu_landyszalay_randomsX50.npy",
        allow_pickle=True,
    ).item()
    s = measured_for_cov['s']
    expected_covariance = measured_for_cov["multipoles"]
    expected_covariance = expected_covariance[..., (s > 30.) & (s<60.)]
    expected_covariance = expected_covariance[:,2,:]
    expected_covariance = np.cov(expected_covariance.reshape((1000,-1)).T)
    hartlap_correction = (1000-1) / (1000 - len(expected_covariance) - 2)
    np.testing.assert_almost_equal(
        hartlap_correction * expected_covariance,
        covariance,
    )


def test_get_covariance_data_combined():
    covariance = Nested.get_covariance_matrix(
        statistics=["density_split_cross", "density_split_auto"],
        select_filters=None,
        slice_filters=None,
        add_emulator_error=False,
        apply_hartlap_correction=True,
    )
    measured_for_cov = np.load(
        "../../data/covariance/ds/gaussian/ds_cross_xi_smu_zsplit_gaussian_Rs10_landyszalay_randomsX50.npy",
        allow_pickle=True,
    ).item()
    expected_covariance_tpcf = measured_for_cov["multipoles"].reshape((1000,-1))
    measured_for_cov = np.load(
        "../../data/covariance/ds/gaussian/ds_auto_xi_smu_zsplit_gaussian_Rs10_landyszalay_randomsX50.npy",
        allow_pickle=True,
    ).item()
    expected_covariance_ds = measured_for_cov["multipoles"].reshape((1000,-1))
    expected_covariance = np.hstack((expected_covariance_tpcf, expected_covariance_ds))
    expected_covariance = np.cov(expected_covariance.T)
    hartlap_correction = (1000-1) / (1000 - len(expected_covariance) - 2)
    np.testing.assert_almost_equal(
        hartlap_correction * expected_covariance,
        covariance,
    )


def test_priors():
    prior_config = {
        'stats_module': 'scipy.stats',
        'a': {'distribution': 'uniform', 'min': 0.01, 'max': 0.02},
        'b': {'distribution': 'uniform', 'min': -0.04, 'max': -0.02},
    }
    priors = Nested.get_priors(
        prior_config=prior_config,
        parameters_to_fit=['a','b'],
    )

    assert priors['a'].rvs() > 0.01
    assert priors['a'].rvs() < 0.02
    assert priors['b'].rvs() > -0.04
    assert priors['b'].rvs() < -0.02


def test_get_theory_model():
    theory_model = Nested.get_theory_model(
        theory_config={
            'module': 'sunbird.summaries',
            'class': 'DensitySplitCross',
        },
    )
    assert isinstance(theory_model, DensitySplitCross)

