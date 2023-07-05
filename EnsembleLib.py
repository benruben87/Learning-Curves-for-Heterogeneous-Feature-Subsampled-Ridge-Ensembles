#Ensembling Library for training ensembles of single-layer networks to make predictions.
#Doesn't use pytorch neural network models since these are overkill for this simple task.
#Adaptable to do linear regression, classification (score-averaging or majority-vote), etc.

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torchvision
import copy

import torch
from torchvision.transforms import Resize

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

from scipy.stats import gamma

tz = pytz.timezone('US/Eastern')
torch.set_default_dtype(torch.float64)

def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)
    
def makeProjectionMatrix(proj_ratio, num_features, align_proj, features = None):
    
    if torch.is_tensor(proj_ratio):
        proj_ratio_reg = proj_ratio.item()
    
    #Later: Try instead learning a smart projection using monte carlo sampling of the projection vector, which consists of zeros and 1's in a fixed ratio.
    k = int(round(proj_ratio_reg*num_features))
    #If align_proj is true, projects to a random selection of neurons
    if align_proj:
        mask = torch.randperm(num_features, device='cuda', requires_grad=False)[:k]
        proj_mat = torch.eye(num_features, device = 'cuda')[mask]
        #proj_mat = torch.zeros(num_features, device = 'cuda')
        #proj_mat[mask]=1
    #If align_proj is false, generates a random gaussian projection matrix.
    else:
        proj_mat = 1/torch.sqrt(torch.tensor(k, device='cuda'))*torch.zeros(size=[k, num_features], device='cuda').normal_(mean=0, std=1)
    return proj_mat

#Makes exclusive A list sequentially.
def makeExclusiveAList(nrs, num_features):
    Sum = 0
    for nr in nrs:
        Sum+=nr
    
    assert Sum<=num_features
    
    eye = torch.eye(num_features).cuda()
    startInd = 0
    A_list = []
    for nr in nrs:
        endInd = startInd+nr
        A_list.append(eye[startInd:endInd,:])
        startInd = startInd+endInd
        
    return A_list

def genNs(K, M, ConnDist = 'delta'):
    alphas = genAlphas(K, ConnDist)
    ns = M*alphas
    #Make sure projections are from 1-M neurons
    ns[ns<1] = 1
    ns[ns>M] = M
    ns = [int(n) for n in ns]
    return ns

#Generates alphas
def genAlphas(K, ConnDist='delta', sigmafrac = 1/2):
    if ConnDist == 'delta':
        return genDeltaAlphas(K)
    elif ConnDist == 'unif':
        return genUnifAlphas(K)
    elif ConnDist == 'exp':
        return genExpAlphas(K)
    elif ConnDist == 'lognorm':
        return genLogNormAlphas(K)
    elif ConnDist == 'gaussian' or ConnDist == 'gauss':
        return genGaussianAlpha(K)
    elif ConnDist == 'gamma' or ConnDist == 'Gamma':
        return genGammaAlphas(K, sigmafrac)
    else:
        raise Exception('Invalid Connectivity Distribution Type.  Acceptable types are delta, unif, exp, lognorm.')

#Generates K projection ratios with lognormal distribution, normalized to sum to 1.
def genLogNormAlphas(K, var = 1):
    ExpectedValue = 1/K
    mu = np.log(1/K)-var/2
    alphas = np.random.lognormal(mean = mu, sigma = var, size = K)
    for i in range(len(alphas)):
        if alphas[i]>1:
            alphas[i] = 1
    alphas = alphas/np.sum(alphas)
    return alphas
        
def genUnifAlphas(K):
    alphas = np.random.uniform(size = K)
    alphas = alphas/np.sum(alphas)
    return alphas
    
def genExpAlphas(K):
    alphas = np.random.exponential(size = K)
    alphas = alphas/np.sum(alphas)
    return alphas

def genDeltaAlphas(K):
    return 1/K*np.ones(K)

#Generates gaussian distriuted alphas which are normalized to sum to one.  Mean of gaussian draws is 1/K and standard deviation is 1/3K
def genGaussianAlpha(K, stdFrac=1/3):
    mean = 1/K*np.ones(K)
    alphas = np.random.normal(1/K, stdFrac*1/(K), size = K)
    alphas = np.abs(alphas)
    alphas = alphas/np.sum(alphas)
    return alphas

def genGammaAlphas(K, stdFrac = 1/2):
    mean = 1/K
    variance = (stdFrac*mean)**2
    
    # Calculate the shape and scale parameters based on mean and variance
    shape = mean ** 2 / variance
    scale = variance / mean

    # Generate random numbers from the gamma distribution
    random_numbers = gamma.rvs(shape, scale=scale, size=K)
    random_numbers = random_numbers/np.sum(random_numbers)
    return random_numbers

def matrix_sqrt(A):
    if torch.max(A)==0:
        return A
    else:
        L,V = torch.linalg.eigh(A)
        DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.T

class MultiReadout():
    def __init__(self, num_features, nrs, sigma_0, eta, singleSiteFunc, align_proj = True, exclusive = False, device = 'cuda'):
        
        #num_classes is the number of readouts.  In this case we treat a single binary readout, so we set num_classes=1 manually.
        num_classes = 1
        self.num_classes = 1
        
        self.device = device
        self.numReadoutReps = len(nrs)
        self.nrs = nrs
        self.num_features = num_features
        self.proj_ratios = torch.tensor(nrs, device = self.device)/num_features
        self.align_proj = True
        self.exclusive = exclusive
        
        #Covariance matrix for presynaptic noise added to inputs.
        self.sigma_0 = sigma_0.to(self.device)
        self.sigma_0_root = matrix_sqrt(self.sigma_0)
            
        #List of postsynaptic scalar noise variances
        self.eta = eta
        
        A_list = [] #List of readout sampling matrices.
        w_list = [] #List of learned weight vectors.
        
        if self.exclusive:
            A_list = makeExclusiveAList(self.nrs, self.num_features)
            w_list = [torch.zeros(nr, 1).to(device) for nr in self.nrs]
        else:
            for i in range(self.numReadoutReps):
                A_list.append(makeProjectionMatrix(self.proj_ratios[i], num_features, align_proj).to(device))
                N = int(self.proj_ratios[i]*num_features)
                w_list.append(torch.zeros(N,1).to(device))
        
        self.A_list = A_list
        self.w_list = w_list
        
        #Nonlinearity (or linearity) associated with individual readouts (identity or sign)
        self.singleSiteFunc = singleSiteFunc
        
    #Utility Function Used to reset the readout structure for a new set of readout ns
    #Presumably faster than initializing a new object.
    def setDegrees(self, nrs):
        self.numReadoutReps = len(nrs)
        self.nrs = nrs
        self.proj_ratios = torch.tensor(nrs, device = self.device)/self.num_features
        
        A_list = [] #List of readout sampling matrices.
        w_list = [] #List of learned weight vectors.
        
        eye = torch.eye(self.num_features).to('cuda')
            
        for i in range(self.numReadoutReps):
            A_list.append(makeProjectionMatrix(self.proj_ratios[i], self.num_features, self.align_proj).to(self.device))
            N = int(self.proj_ratios[i]*self.num_features)
            w_list.append(torch.zeros(N,1).to(self.device))
        
        self.A_list = A_list
        self.w_list = w_list
    
    def get_A_List():
        return self.A_list
    
    def forward(self, x, ensFunc):
        
        #x is taken in to have rows as the data. 
        standardNormal = torch.normal(torch.zeros(self.num_features), torch.ones(self.num_features)).to(self.device)
        popNoise = self.sigma_0_root@standardNormal
        
        #Add presynaptic noise to features.
        features = x + popNoise
        
        #Project features to ensemble readouts and add additional presynaptic noise (due to timing variations in readout from each neuron.)
        features_proj = [features @ self.A_list[i].T for i in range(self.numReadoutReps)]
        
        rawOutputs =  torch.stack([(1/(self.nrs[i]**(1/2))*features_proj[i]@self.w_list[i] + self.eta*torch.normal(torch.zeros(1), torch.ones(1)).to(self.device)).squeeze() for i in range(self.numReadoutReps)])
        
        singleSiteOutputs = self.singleSiteFunc(rawOutputs)
        ensembleOutput = ensFunc(rawOutputs)
        
        return rawOutputs, singleSiteOutputs, ensembleOutput
    

        
    #Method to fit only the readout weights using linear regression.
    #Assumes training loader and test loader store a single (full) batch
    def regress_readout(self, tr_loader, test_loader, lam = 0):
        
        N_tr = tr_loader[0].shape[0]
        shared_noise = self.sigma_0_root@torch.normal(torch.zeros(self.num_features, N_tr), torch.ones(self.num_features, N_tr)).to(self.device)
        
        for i in range(self.numReadoutReps):
            
            psi = (self.nrs[i])**(-1/2)*self.A_list[i]@(tr_loader[0].T+shared_noise)# Nr by P matrix of features
            psiT = psi.T
            Y = tr_loader[1] + self.eta*torch.normal(torch.zeros(N_tr), torch.ones(N_tr)).to(self.device) #Add readout noise during training as well!
            #if features.shape[0]>features.shape[1]:
            #    W = targets@torch.linalg.pinv(features_T@features)@features_T
            #else:
            #    W = targets@features_T@torch.linalg.pinv(features@features_T)
            
            if lam<0:
                raise Exception('Regularization parameter must be non-negative')
            elif lam==0:
                W = torch.linalg.pinv(psiT)@Y
            else:
                P = psi.shape[1]
                Nr = psi.shape[0]
                if P<=Nr:
                    W = psi@torch.linalg.solve(psiT@psi + lam*torch.eye(P, device = 'cuda'), Y)#psi @ torch.linalg.inv(psiT@psi + lam*torch.eye(P, device = 'cuda'))@Y
                else:
                    W = torch.linalg.solve((psi@psiT + lam*torch.eye(Nr, device = 'cuda')), psi@Y)#torch.linalg.inv(psi@psiT + lam*torch.eye(Nr, device = 'cuda'))@psi@Y
                    
                 
            self.w_list[i] = W.reshape(-1)
        
        #model_err = self.eval(tr_loader, test_loader)

    #Evaluates the training and test error of the model
    #ToDo: Modify to calculate training loss, test losos, and single-site training and test errors.
    def eval(self, tr_loader, test_loader, ensFunc, errFunc):
    
        #Calculate Trainint Error
        x = tr_loader[0]
        y = tr_loader[1]
        rawOut, out_reps, out = self.forward(x, ensFunc)
        err_train = errFunc(out, y)
                
        x = test_loader[0]
        y = test_loader[1]
        rawOut_reps, out_reps, out = self.forward(x, ensFunc)
        err_test = errFunc(out,y) #1/4*torch.mean(torch.square(out-y))
        
        return (err_train, err_test)
    
#     def calcLoss(self, tr_loader, training=False):
#         self.model.eval()
#         with torch.no_grad():
#             with autocast():
#                 # Predict and compute loss
#                 y_preds meanYPred = self.forward(tr_loader[0])
#                 loss = 0
#                 for i in range(self.numReadoutReps):
#                     loss += torch.mean(torch.square(y_preds[i] - tr_loader[1])).detach().cpu().numpy()
#                 #test_loss = self.calcEg()
#             self.model.train()
#         return (loss, test_loss)
    
    #Uses the analytical formula for the generalization error in the case of a linear ensemble of predictors.
    def calcEg_linear(self, sigma_s, w_star):
        
        Eg = 0
        w_star = w_star.reshape(-1,1)
        
        #Diagonal Terms
        for r in range(self.numReadoutReps):
            w_r = self.w_list[r].reshape(-1,1)
            A_r = self.A_list[r]   
            M = self.num_features
            K = self.numReadoutReps
            
            sigma_0 = self.sigma_0
            Nr = self.nrs[r]
            alpha_r = torch.tensor([Nr/M], device = self.device)
            eta_r = self.eta
            
            curE = 1/M*((1/torch.sqrt(alpha_r)*A_r.T@w_r - w_star).T @ sigma_s @ (1/torch.sqrt(alpha_r)*A_r.T@w_r - w_star) + 1/alpha_r*w_r.T@A_r@sigma_0@A_r.T@w_r) + eta_r*eta_r
            Eg+=curE
            
        
        for r in range(self.numReadoutReps):
            for rp in range(r+1, self.numReadoutReps):
                
                w_r = self.w_list[r].reshape(-1,1)
                w_rp = self.w_list[rp].reshape(-1,1)
                A_r = self.A_list[r]
                A_rp = self.A_list[rp]
                M = self.num_features
                K = self.numReadoutReps

                sigma_0 = self.sigma_0
                
                Nr = self.nrs[r]
                Nrp = self.nrs[rp]
                alpha_r = torch.tensor([Nr/M], device = self.device)
                alpha_rp = torch.tensor([Nrp/M], device = self.device)
                eta_r = self.eta
                eta_rp = self.eta
                
                curE = 2/M*((1/torch.sqrt(alpha_r)*A_r.T@w_r - w_star).T @ sigma_s @ (1/torch.sqrt(alpha_rp)*A_rp.T@w_rp - w_star) + 1/torch.sqrt(alpha_r*alpha_rp)*w_r.T@A_r@sigma_0@A_rp.T@w_rp)
                
                Eg += curE
                
        return 1/(K**2) * Eg