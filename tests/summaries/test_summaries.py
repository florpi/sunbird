import torch
from pathlib import Path
import json
import pytest
import numpy as np
import xarray as xr
from sunbird.summaries.base import BaseSummary
from sunbird.data.transforms import Transforms, Normalize, LogSqrt
from sunbird.data import AbacusDataModule

# from sunbird.summaries.tpcf import TPCF
# from sunbird.summaries.density_split import DensitySplitBase

class DummyModel:
    def __call__(self, inputs,):
        return inputs**2

@pytest.fixture
def bs():
    return BaseSummary(
        DummyModel(),
        coordinates={
            'a': np.arange(10),
            'b': np.arange(20),
        },
        input_names=['a','b'],
    )

def test__base_summary(bs):
    inputs = np.random.random((10,20))
    np.testing.assert_allclose(bs.forward(inputs), (inputs**2).reshape(-1), rtol=1.e-5)

def test__base_summary_with_filters(bs):
    inputs = np.random.random((10,20))
    select_filters = {'b': [0,1],}
    slice_filters = {'a': [0,5]}
    expected = inputs[:6, :2]**2
    np.testing.assert_allclose(bs.forward(
        inputs, 
        select_filters=select_filters, 
        slice_filters=slice_filters
    ), expected.reshape(-1), rtol=1.e-5)

def test__base_summary_from_folder():
    bs = BaseSummary.from_folder(
        path_to_model="../../trained_models/test/version_5/",
        flax=False,
    )
    inputs = np.random.random((len(bs.input_names)))
    inputs = dict(zip(bs.input_names, inputs))
    bs_output = bs(inputs)
    assert bs_output.shape == (72,)


def test__base_summary_from_folder_batched():
    bs = BaseSummary.from_folder(
        path_to_model="../../trained_models/test/version_5/",
        flax=False,
    )
    inputs = np.random.random((10,len(bs.input_names)))
    bs_output = bs.get_for_batch_inputs(inputs, select_filters={'multipoles':[0,]})
    assert bs_output.shape == (10,36)
    
def test__base_summary_flax():
    bs = BaseSummary.from_folder(
        path_to_model="../../trained_models/test/version_5/",
        flax=False,
    )
    bs_flax = BaseSummary.from_folder(
        path_to_model="../../trained_models/test/version_5/",
        flax=False,
    )
    inputs = np.random.random((len(bs.input_names)))
    params = dict(zip(bs.input_names, inputs))
    bs_output = bs(params, select_filters={'multipoles':[0,]})
    bs_flax_output = np.array(bs_flax(params, select_filters={'multipoles':[0,]}))
    np.testing.assert_allclose(bs_output, bs_flax_output, rtol=1e-5, atol=1e-5)

def test__base_summary_with_transforms():
    select_filters = {
        "quintiles": [4],
        "multipoles": [0,2,],
    }
    log_sqrt = LogSqrt()
    norm = Normalize(dimensions=['realizations', 'cosmology', 'quintiles', 's'])
    ds = AbacusDataModule(
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3,4],},
        output_transforms=Transforms([log_sqrt,norm]),
        select_filters=select_filters,
    )
    ds.setup(stage="train")
    path_to_data = Path(__file__).parent.parent.parent / "data/"
    with open(path_to_data / f'coordinates/density_split_cross.json') as fd:
        coordinates = json.load(fd)
    coordinates = {
            'multipoles': [0,2],
            's': coordinates['s'][1:],
    }
    transformed_data = ds.ds_val.tensors[1][0]
    transformed_data = transformed_data.reshape((2, 36))
    transformed_data = xr.DataArray(
        transformed_data,
        coords= coordinates,
        dims=['multipoles', 's'],
    )
    bs = BaseSummary(
        DummyModel(),
        coordinates=coordinates,
        input_transforms=ds.input_transforms,
        output_transforms=ds.output_transforms,
    )

    recovered_data = bs.output_transforms.inverse_transform(transformed_data)
    true_data = ds.data.read_statistic(
        cosmology=3,
        statistic='density_split_cross',
        phase=0,
    )[0].sel(quintiles=4)
    transformed_true_data = ds.output_transforms.transform(true_data)
    recovered_true_data = ds.output_transforms.inverse_transform(transformed_true_data)
    np.testing.assert_allclose(transformed_true_data.values, transformed_data, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(recovered_true_data.values, true_data.values, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(recovered_data.values, true_data.values, rtol=1e-5, atol=1e-5)

        
"""


"""
"""
def test__tpcf():
    tpcf = TPCF()
    inputs = np.random.random((len(tpcf.parameters)))
    params = dict(zip(tpcf.parameters, inputs))
    assert tpcf(params).shape == (len(tpcf.s),)


def test__density_split():
    ds = DensitySplitBase()
    inputs = np.random.random((len(ds.parameters)))
    params = dict(zip(ds.parameters, inputs))
    assert ds(params).shape == (len(ds.s)*len(ds.quintiles),)


def test__tpcf_batch():
    tpcf = TPCF()
    inputs = np.random.random((10,len(tpcf.parameters)))
    params = dict(zip(tpcf.parameters, inputs.T))
    assert tpcf.get_for_batch(params).shape == (10,len(tpcf.s),)


def test__density_split_catch():
    ds = DensitySplitBase()
    inputs = np.random.random((10,len(ds.parameters)))
    params = dict(zip(ds.parameters, inputs.T))
    assert ds.get_for_batch(params).shape == (10,len(ds.s)*len(ds.quintiles),)
"""
