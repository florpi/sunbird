from sunbird.data import AbacusDataModule
import torch


def test__normalize_inputs():
    ds = AbacusDataModule(
        train_test_split_dict={"train": range(130,170), "test": [2], 'val': [3],},
        normalize_inputs=True,
    )
    ds.setup(stage="train")
    assert all(torch.min(ds.ds_train.tensors[0], dim=0).values == 0.0)
    assert all(torch.max(ds.ds_train.tensors[0], dim=0).values == 1.0)


def test__normalize_outputs():
    ds = AbacusDataModule(
        train_test_split_dict={"train": [0,1], "test": [2], 'val': [3],},
        normalize_outputs=True,
    )
    ds.setup(stage="train")
    assert float(torch.min(ds.ds_train.tensors[1])) == 0.0
    assert float(torch.max(ds.ds_train.tensors[1])) == 1.0


def test__standarize_outputs():
    ds = AbacusDataModule(
        train_test_split_dict={"train": [0,1], "test": [2], 'val': [3],},
        standarize_outputs=True,
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
