import xarray as xr
import numpy as np
from sunbird.data.transforms import LogSqrt

def test_inverse_works():
    dr = xr.DataArray(
        np.random.random((10, 20)),
        coords = {
            'multipoles': np.arange(10),
            'b': np.arange(20),
        },
    )
    logsqrt = LogSqrt()
    recovered_dr = logsqrt.inverse_transform(logsqrt.transform(dr))

    np.testing.assert_allclose(dr.values, recovered_dr.values)