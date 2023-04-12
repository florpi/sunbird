import torch
import pytest
import argparse
from sunbird.data.transforms import Transforms, Normalize, Standarize
from sunbird.data import AbacusDataModule


def test__normalize_inputs():
    ds = AbacusDataModule(
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
    )
    ds.setup(stage="train")
    assert all(torch.min(ds.ds_train.tensors[0], dim=0).values == 0.0)
    assert all(torch.max(ds.ds_train.tensors[0], dim=0).values == 1.0)


def test__normalize_outputs():
    ds = AbacusDataModule(
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
    )
    ds.setup(stage="train")
    assert float(torch.min(ds.ds_train.tensors[1])) == 0.0
    assert float(torch.max(ds.ds_train.tensors[1])) == 1.0

def test__normalize_outputs_by_multipole():
    select_filters = {
        "quintiles": [4],
        "multipoles": [0,2,],
    }
    norm = Normalize(dimensions=['realizations', 'cosmology', 'quintiles', 's'])
    ds = AbacusDataModule(
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
        output_transforms=Transforms([norm]),
        select_filters=select_filters,
    )
    ds.setup(stage="train")

    assert float(torch.min(ds.ds_train.tensors[1][:, :36])) == 0.0
    assert float(torch.max(ds.ds_train.tensors[1][:, :36])) == 1.0

    assert float(torch.min(ds.ds_train.tensors[1][:, 36:])) == 0.0
    assert float(torch.max(ds.ds_train.tensors[1][:, 36:])) == 1.0



def test__standarize_outputs():
    ds = AbacusDataModule(
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
        output_transforms=Transforms([Standarize(dimensions=['cosmology', 'realizations'])]),
    )
    ds.setup(stage="train")
    mean = torch.mean(ds.ds_train.tensors[1], dim=0)
    std = torch.std(ds.ds_train.tensors[1], dim=0)

    torch.allclose(
        mean,
        torch.ones_like(mean),
    )
    torch.allclose(
        std,
        torch.zeros_like(mean),
    )

class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def test_from_argparse_args():
    parser = argparse.ArgumentParser()
    parser = AbacusDataModule.add_argparse_args(parser)
    args = DummyArgs(**{
        'statistic': 'density_split_cross',
        'select_quintiles': [0,],
        'select_multipoles': [0,2],
        'slice_s': [0.7, 150.],
        'batch_size' : 32,
        'abacus_dataset': 'wideprior_AB',
        'input_parameters': None,
        'independent_avg_scale': False,
        'input_transforms': ['Normalize'],
        'output_transforms': ['Normalize'],
    }
    )
    dm = AbacusDataModule.from_argparse_args(
        args, 
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
    )
    dm.setup(stage="train")

    assert all(torch.min(dm.ds_train.tensors[0], dim=0).values == 0.0)
    assert all(torch.max(dm.ds_train.tensors[0], dim=0).values == 1.0)

    assert float(torch.min(dm.ds_train.tensors[1][:,:36])) == 0.0
    assert float(torch.max(dm.ds_train.tensors[1][:,:36])) == 1.0

    assert float(torch.min(dm.ds_train.tensors[1][:,36:])) == 0.0
    assert float(torch.max(dm.ds_train.tensors[1][:,36:])) == 1.0

def test_from_argparse_args_independent_scale():
    parser = argparse.ArgumentParser()
    parser = AbacusDataModule.add_argparse_args(parser,)
    args = DummyArgs(**{
        'statistic': 'density_split_cross',
        'select_quintiles': [0,],
        'select_multipoles': [0,2],
        'slice_s': [0.7, 150.],
        'batch_size' : 32,
        'abacus_dataset': 'wideprior_AB',
        'input_parameters': None,
        'independent_avg_scale':  True,
        'input_transforms': ['Normalize'],
        'output_transforms': ['Normalize'],

    }
    )
    dm = AbacusDataModule.from_argparse_args(
        args, 
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
    )
    dm.setup('train')
    multipoles = dm.ds_train.tensors[1].reshape((-1, 2, 36))

    assert all(torch.min(multipoles[:,0,:], axis=0).values == 0.0)
    assert all(torch.max(multipoles[:,0,:], axis=0).values == 1.0)

    assert all(torch.min(multipoles[:,1,:], axis=0).values == 0.0)
    assert all(torch.max(multipoles[:,1,:], axis=0).values == 1.0)
