import glob
import sys
import h5py
import dynesty
import torch 
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from sunbird.emulators import FCN
from scipy import spatial,stats
from torch import set_num_threads
set_num_threads(1)
import os
os.environ["OMP_NUM_THREADS"] = "1"
Rg = int(sys.argv[1])
Nm   = int(sys.argv[2])

def mfs_model(Rg,Nm):
    root_dir = '/data/wliu/home/DESIacm/EMC/85_cosmos/Emulator_Nur/MFs/Rg'+str(Rg)+'_Bins60_80_100_120/'
    Model_fns = np.array(glob.glob(root_dir+'*'))
    val_loss = np.array([np.float64(fn.split('/')[-1].split('loss=')[-1][:-5]) for fn in Model_fns])
    Model_chosen = Model_fns[np.argsort(val_loss)[Nm]]
    epoch = int(Model_fns[np.argsort(val_loss)[Nm]].split('/')[-1].split('epoch=')[-1].split('-')[0])
    model = FCN.load_from_checkpoint(Model_chosen, strict=True)
    model.eval()
    return model,epoch,val_loss[np.argsort(val_loss)[Nm]]

def read_trainset(Rg):
    Rmax = Rg + 5
    f = h5py.File('/data/wliu/home/DESIacm/EMC/85_cosmos/Dataset_Nur/trainset_mfs_rg'+str(Rg)+'_Bins60_80_100_120.hdf5','r')
    Params = f['model_params'][:,:]
    MFs = f["model_mfs"][:,:]
    cov_mfs = f['cov_mfs'][:,:]
    return Params,MFs,cov_mfs

def read_testset(Rg):
    Rmax = Rg + 5
    f = h5py.File('/data/wliu/home/DESIacm/EMC/85_cosmos/Dataset_Nur/testset_mfs_rg'+str(Rg)+'_Bins60_80_100_120.hdf5','r')
    Params = f['model_params'][:,:]
    MFs = f["model_mfs"][:,:]
    diffsky_mfs = f["diffsky_mfs_unit0"][:]
    return Params,MFs,diffsky_mfs

lhc_train_x, lhc_train_y, cov_mfs    = read_trainset(Rg)
lhc_test_x,  lhc_test_y, diffsky_mfs = read_testset(Rg)

train_y_mean = np.mean(lhc_train_y, axis=0)
train_y_std  = np.std(lhc_train_y, axis=0)
train_x_mean = np.mean(lhc_train_x, axis=0)
train_x_std  = np.std(lhc_train_x, axis=0)

def transform_x(x):
    return (x-train_x_mean)/(train_x_std)

def transform_y(y):
    return (y-train_y_mean)/(train_y_std)

def inverse_transform_x(x):
    return x*train_x_std+train_x_mean

def inverse_transform_y(y):
    return y*train_y_std+train_y_mean


mf_emu,epoch,val_loss = mfs_model(Rg,Nm)
with torch.no_grad():
    pred_test_y  = mf_emu.get_prediction(torch.Tensor(transform_x(lhc_test_x)))
pred_test_y = pred_test_y.numpy()
pred_test_y = inverse_transform_y(pred_test_y)
C_emu = np.cov(np.transpose((pred_test_y - lhc_test_y)))
Cinv = np.linalg.inv(np.cov(cov_mfs.T)/8+C_emu) 
miu = diffsky_mfs


def get_y(param):
    with torch.no_grad():
        x = mf_emu.get_prediction(param)
    x = x.detach().numpy()
    return inverse_transform_y(x)
    
def loglike(param):
    """The log-likelihood function."""
    x = get_y(torch.Tensor(transform_x(param)))

    return -0.5 * np.dot(x-miu, np.dot(Cinv, x-miu))

# Define our uniform prior.
def ptform(u):
    x = np.array(u)  # copy u
    #####################omega_b,omega_cdm,sigma_8, n_s, alpha_s, N_eff, w0_fld, wa_fld, logM_cut,logM_1,sigma, alpha,kappa,alpha_c,alpha_s,  s,  A_cen, A_sat,B_cen,B_sat
    param_min = np.array([0.0207,  0.1032,  0.678, 0.901, -0.038, 1.177, -1.27, -0.628, 12.5,    13.6, -2.99, 0.3,  0.,    0.,    0.58,  -0.98,-0.99,   -1.,-0.67,-0.97])
    param_max = np.array([0.0243,  0.14,    0.938, 1.025,  0.038, 2.889, -0.70,  0.621, 13.7,    15.1,  0.96, 1.48, 0.99,  0.61,   1.49,    1.,  0.93,    1., 0.2,  0.99])
    x = u*(param_max-param_min) + param_min
    x = u*(param_max-param_min) + param_min
    #For omega_b
    t = stats.norm.ppf(u[0])  # convert to standard normal
    x[0] = t*0.00038
    x[0] +=  0.02237  ###for fit of BOSS data we should use 0.02268

    return x

root_dir = '/data/wliu/home/DESIacm/EMC/85_cosmos/Inference/MFs/Rg'+str(Rg)+'_Bins60_80_100_120/'
ndim = 20
post_fix = 'DiffskyUnit_FitAll_epoch'+str(epoch)+'_loss'+str(val_loss)

with Pool(64) as pool:
    dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim, pool=pool,queue_size=64,use_pool={'prior_transform': False}, nlive=2048)
    dsampler.run_nested(checkpoint_file=root_dir+'Results/'+post_fix+'.save',maxiter=2000000, maxcall=10000000)
    res = dsampler.results
    res.samples.tofile(root_dir+'Samples/'+post_fix+'.bin')
    res.logl.tofile(root_dir+'Likelihoods/'+post_fix+'.bin')
    weight = np.exp(res.logwt)
    weight.tofile(root_dir+'Weights/'+post_fix+'.bin')