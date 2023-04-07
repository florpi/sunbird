from sunbird.data.data_readers import Abacus, CMASS, Uchuu


def test__get_path():
    dr = Abacus()
    assert (
        dr.get_file_path(statistic="tpcf", cosmology=118, phase=0).name
        == "tpcf_c118_ph000.npy"
    )


def test__get_observation_abacus():
    dr = Abacus(
        statistics=["tpcf", "density_split_auto", "density_split_cross"],
        select_filters={"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]},
    )
    assert dr.get_observation(
        cosmology=0,
        hod_idx=100,
        phase=0,
    ).shape == (36 * 2 * 9,)


def test__get_observation_cmass():
    dr = CMASS(
        statistics=["tpcf", "density_split_auto", "density_split_cross"],
        select_filters={"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]},
    )
    assert dr.get_observation().shape == (36 * 2 * 9,)


def test__get_observation_uchuu():
    dr = Uchuu(
        statistics=["tpcf", "density_split_auto", "density_split_cross"],
        select_filters={"multipoles": [0, 2], "quintiles": [0, 1, 3, 4]},
    )
    assert dr.get_observation(ranking="random").shape == (36 * 2 * 9,)
