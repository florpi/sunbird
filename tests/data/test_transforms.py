import xarray as xr
import pytest
import numpy as np
from sunbird.data.transforms import LogSqrt, Normalize, Standarize, Transforms

@pytest.fixture
def dr():
    return xr.DataArray(
        10.*np.random.random((10, 20)) + 1.,
        coords = {
            'multipoles': np.arange(10),
            'b': np.arange(20),
        },
    )

@pytest.mark.parametrize("transform", [LogSqrt(), Normalize(), Standarize()])
def test_inverse_works(transform, dr):
    if hasattr(transform, 'fit'):
        transform.fit(dr,)
    recovered_dr = transform.inverse_transform(transform.transform(dr))
    np.testing.assert_allclose(dr.values, recovered_dr.values)

def test_transfom_over_dimension(dr):
    norm = Normalize(dimensions=['b'])
    normed_dr = norm.fit_transform(dr)
    assert normed_dr[0,:].min() == 0.
    assert normed_dr[0,:].max() == 1.
    assert normed_dr[1,:].min() == 0.
    assert normed_dr[1,:].max() == 1.


def test_combine_transforms(dr):
    logsqrt = LogSqrt()
    norm = Normalize(training_min=dr.min(), training_max=dr.max())
    transforms = Transforms([logsqrt, norm])
    tranformed_dr = transforms.transform(dr)
    should_dr =  norm.transform(logsqrt.transform(dr))
    np.testing.assert_allclose(tranformed_dr.values, should_dr.values)

def test_store_and_load_transforms(dr):
    logsqrt = LogSqrt()
    norm = Normalize()
    transforms = Transforms([logsqrt, norm])
    dr = transforms.fit_transform(dr, path_to_store='test.pkl')
    recovered_transforms = Transforms.from_file('test.pkl')
    assert len(transforms.transforms) == len(recovered_transforms.transforms)
    assert isinstance(transforms.transforms[0], LogSqrt)
    assert isinstance(transforms.transforms[1], Normalize)

    np.testing.assert_allclose(
        transforms.transforms[1].training_min.values,
        norm.training_min.values,
    )
