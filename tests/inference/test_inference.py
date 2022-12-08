import numpy as np
from sunbird.inference import Nested 

def test_observation():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology = 124,
        hod_idx = 4,
        statistics = ['density_split_cross'],
        select_filters=None,
        slice_filters=None
    )
    path_direct = '../../data/clustering/different_hods/ds/gaussian/ds_cross_xi_smu_zsplit_Rs20_c124_ph000.npy'
    observation_direct = np.load(path_direct, allow_pickle=True).item()
    s = observation_direct['s']
    observation_direct = np.mean(observation_direct['multipoles'][4],axis=0)
    np.testing.assert_almost_equal(
        observation_from_inference,
        observation_direct.reshape(-1),
    )

def test_observation_with_filters():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology = 124,
        hod_idx = 4,
        statistics = ['density_split_cross'],
        select_filters={'multipoles': [2], 'quintiles': [1,4]},
        slice_filters={'s': [30.,60.]}
    )
    path_direct = '../../data/clustering/different_hods/ds/gaussian/ds_cross_xi_smu_zsplit_Rs20_c124_ph000.npy'
    observation_direct = np.load(path_direct, allow_pickle=True).item()
    s = observation_direct['s']
    observation_direct = np.mean(observation_direct['multipoles'][4],axis=0)
    observation_direct = observation_direct[..., (s>30.) & (s<60.)]
    observation_direct = observation_direct[[1,4],:, :]
    observation_direct = observation_direct[:,2, :]
    np.testing.assert_almost_equal(
        observation_from_inference,
        observation_direct.reshape(-1),
    )

def test_observation_combined_summaries():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology = 124,
        hod_idx = 4,
        statistics = ['density_split_auto', 'density_split_cross'],
        select_filters=None,
        slice_filters=None
    )
    observations = []
    for corr in ['auto', 'cross']:
        path_direct = f'../../data/clustering/different_hods/ds/gaussian/ds_{corr}_xi_smu_zsplit_Rs20_c124_ph000.npy'
        observation_direct = np.load(path_direct, allow_pickle=True).item()
        s = observation_direct['s']
        observations.append(np.mean(observation_direct['multipoles'][4],axis=0))
    observations = np.array(observations)
    np.testing.assert_almost_equal(
        observation_from_inference,
        observations.reshape(-1),
    )

def test_observation_with_filters_combined():
    observation_from_inference = Nested.get_observation_for_abacus(
        cosmology = 124,
        hod_idx = 4,
        statistics = ['density_split_auto', 'density_split_cross'],
        select_filters={'multipoles': [2], 'quintiles': [1,4]},
        slice_filters={'s': [30.,60.]},
    )
    observations = []
    for corr in ['auto', 'cross']:
        path_direct = f'../../data/clustering/different_hods/ds/gaussian/ds_{corr}_xi_smu_zsplit_Rs20_c124_ph000.npy'
        observation_direct = np.load(path_direct, allow_pickle=True).item()
        s = observation_direct['s']
        observation_direct = np.mean(observation_direct['multipoles'][4],axis=0)
        observation_direct = observation_direct[..., (s>30.) & (s<60.)]
        observation_direct = observation_direct[[1,4],:, :]
        observation_direct = observation_direct[:,2, :]
        observations.append(
           observation_direct 
        )
    observations = np.array(observations)
    np.testing.assert_almost_equal(
        observation_from_inference,
        observations.reshape(-1),
    )