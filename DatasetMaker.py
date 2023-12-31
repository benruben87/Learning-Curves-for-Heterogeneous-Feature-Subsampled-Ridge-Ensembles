#Library used to make datasets for regression and classification experiments.
import numpy as np
from collections import OrderedDict

import torch
import torchvision
import copy
from torchvision.transforms import Resize
torch.set_default_dtype(torch.float64)

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

tz = pytz.timezone('US/Eastern')
torch.set_default_dtype(torch.float64)

def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def matrix_sqrt(A):
    L,V = torch.linalg.eigh(A)
    DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.T

def split_data(P, x, y):
    indices = np.random.permutation(x.shape[0])
    trainIndices = indices[0:int(P)]
    testIndices = indices[int(P):]
    
    train_loader = (x[trainIndices], y[trainIndices])
    test_loader = (x[testIndices], y[testIndices])
    return (train_loader, test_loader)

def sample_data(P, x, y, random = False):
    assert P <= x.shape[0]
    if random:
        indices = np.random.permutation(x.shape[0])
        trainIndices = indices[0:int(P)]
        data_loader = (x[trainIndices], y[trainIndices])
    else:
        data_loader = (x[:int(P)], y[:int(P)])
    return data_loader

#Shuffles the order of the data in x and corresponding labels in y.  Returns x and y.
def shuffle_data(x, y):
    assert x.shape[0] == y.shape[0] #make sure data and labels have samve size.
    indices = np.random.permutation(x.shape[0])
    return x[indices], y[indices]

#N is the number of examples in the dataset
#w_star is the ground truth weights
#sigma_s is the covariance matrix of the data generating distribution
#zeta is the scale of label noise
#C is number of output classes
def makeGaussianDataset_lin(N, w_star, sigma_s, zeta = 0):
    sigma_s_root = matrix_sqrt(sigma_s)
    input_dim = sigma_s.shape[0]
    x = torch.normal(torch.zeros(N, input_dim), torch.ones(N, input_dim)).to('cuda') @ sigma_s_root
    y = 1/(input_dim**(1/2))*x @ w_star + zeta*torch.normal(torch.zeros(N), torch.ones(N)).to('cuda')
    y = y.reshape(-1,1)
    return (x.to('cuda'), y.to('cuda'))

def makeGaussianDataset_sgn(N, w_star, sigma_s, zeta = 0):
    sigma_s_root = matrix_sqrt(sigma_s)
    input_dim = sigma_s.shape[0]
    x = torch.normal(torch.zeros(N, input_dim), torch.ones(N, input_dim)).to('cuda') @ sigma_s_root
    y = torch.sign(1/(input_dim**(1/2))*x @ w_star + zeta*torch.normal(torch.zeros(N), torch.ones(N)).cuda()).to('cuda')
    return (x.to('cuda'), y.to('cuda'))

def makeGaussianDataset_class(N, w_star, sigma_s, zeta = 0):
    sigma_s_root = matrix_sqrt(sigma_s)
    input_dim = sigma_s.shape[0]
    x = torch.normal(torch.zeros(N, input_dim), torch.ones(N, input_dim)).to('cuda') @ sigma_s_root
    y = torch.sign(1/(input_dim**(1/2))*x @ w_star + zeta*torch.normal(torch.zeros(N), torch.ones(N)).cuda()).to('cuda')
    y = torch.nn.functional.one_hot(torch.abs((y+1)/2).to(torch.int64))
    return (x.to('cuda'), y.to('cuda'))

def makeDenseDisc_Dataset(dat, NMax=0, zscore = False):
    if zscore:
        # subtract off spont PCs
        sresp = (dat['sresp'].copy() - dat['mean_spont'][:,np.newaxis]) / dat['std_spont'][:,np.newaxis]
    else:
        sresp = dat['sresp'].copy()
    x = dat['sresp'].T
    if NMax>0 and NMax<x.shape[1]:
        x = x[:, :NMax]
    y = np.sign(dat['istim']-np.pi/4)# separate above 45 degrees from below 45 degrees    
    return (torch.tensor(x).float().to('cuda'), torch.tensor(y).float().to('cuda'))

def makeEasyDisc_Dataset(dat, NMax=0, zscore = False, theta_pref = 0, thmax = 80*np.pi/180, thmin = 10*np.pi/180):
    
    istim = dat['istim']
    dy = istim - theta_pref
    dy = dy%(2*np.pi)
    dy[dy>np.pi] = dy[dy>np.pi] - 2*np.pi
    include = np.logical_and(np.abs(dy) < thmax, np.abs(dy) > thmin)
    
    if zscore:
        # subtract off spont PCs
        sresp = (dat['sresp'].copy() - dat['mean_spont'][:,np.newaxis]) / dat['std_spont'][:,np.newaxis]
    else:
        sresp = dat['sresp'].copy()
    
    x = dat['sresp'].T
    
    assert(NMax>=0 and NMax<=x.shape[1])
    
    if NMax>0 and NMax<x.shape[1]:
        total_columns = x.shape[1]

        # Generate M unique random indices in the range of the column indices
        rand_column_indices = np.random.choice(total_columns, size=NMax, replace=False)

        # Select these columns from the array
        x = x[:, rand_column_indices]
        
    y = np.sign(dy)# separate positive from negative.
    #Returns only the examples which should be included for the easy task.
    return (torch.tensor(x[include]).float().to('cuda'), torch.tensor(y[include]).float().to('cuda'))