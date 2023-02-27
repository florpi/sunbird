import pytest
from scipy.stats import multivariate_normal
import torch
import numpy as np
from sunbird.models.loss import GaussianNLoglike 

def test__gaussian():
    gl = GaussianNLoglike.from_statistics(
        statistics=['tpcf'],
        slice_filters={'s': [0.7,150.]},
        select_filters={'multipoles': [0,]}
    )
    inputs = torch.ones((len(gl.covariance)), dtype=torch.float32)
    expected = multivariate_normal.pdf(inputs, mean=inputs, cov=gl.covariance)
    found = float(gl(inputs.reshape(1,-1), inputs.reshape(1,-1)))
    found += 0.5*torch.logdet(gl.covariance) + 0.5*len(gl.covariance)*np.log(2*np.pi)
    assert found == pytest.approx(-float(np.log(expected)))

def test__gaussian_with_batch():
    gl = GaussianNLoglike.from_statistics(
        statistics=['tpcf'],
        slice_filters={'s': [0.7,150.]},
        select_filters={'multipoles': [0,]},
    )
    n_dim = gl.covariance.shape[0]
    n_batch = 2
    inputs = torch.tensor(np.random.random(size=(n_batch, n_dim)), dtype=torch.float32)
    targets = torch.tensor(np.random.random(size=(n_batch, n_dim)), dtype=torch.float32)
    loglike = 0
    for batch in range(len(inputs)):
        loglike += gl(inputs[batch].reshape(1,-1), targets[batch].reshape(1,-1))
    assert gl(inputs, targets)  == pytest.approx(loglike/len(inputs))