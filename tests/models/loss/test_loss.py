from sunbird.models.loss import GaussianLoglike 
import pytest
from scipy.stats import multivariate_normal

import torch
import numpy as np

def test__gaussian():
    gl = GaussianLoglike.from_file()
    inputs = torch.ones((len(gl.covariance)))
    expected = multivariate_normal.pdf(inputs, mean=inputs, cov=gl.covariance)
    assert float(gl(inputs, inputs)) == pytest.approx(float(np.log(expected)))

def test__gaussian_with_batch():
    gl = GaussianLoglike.from_file()
    n_dim = gl.covariance.shape[0]
    n_batch = 32
    inputs = torch.tensor(np.random.random(size=(n_batch, n_dim)))
    targets = torch.tensor(np.random.random(size=(n_batch, n_dim)))
    loglike = 0
    for batch in range(len(inputs)):
        loglike += gl(inputs[batch], targets[batch])
    assert gl(inputs, targets)  == pytest.approx(loglike/len(inputs))