import numpy as np
import torch
from sunbird.summaries import TPCF, DensitySplitCross, Bundle

def test__bundle():
    tpcf = TPCF()
    ds = DensitySplitCross()
    bundle = Bundle(['tpcf', 'density_split_cross']) 
    inputs = torch.tensor(np.random.random((10,15)), dtype=torch.float32)
    bundle_output = bundle.forward(inputs,None, None)
    bundle_output = bundle_output.detach().numpy()
    tpcf_output = tpcf.forward(inputs,None, None).detach().numpy()
    ds_output = ds.forward(inputs,None, None).detach().numpy()
    np.testing.assert_almost_equal(
        bundle_output[:, :tpcf_output.shape[-1]],
        tpcf_output,
    )
    np.testing.assert_almost_equal(
        bundle_output[:, tpcf_output.shape[-1]:],
        ds_output.reshape((len(ds_output),-1)),
    )
