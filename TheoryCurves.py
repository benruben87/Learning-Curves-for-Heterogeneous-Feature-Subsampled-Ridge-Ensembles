import numpy as np
import time
from datetime import datetime
import copy
from tqdm import tqdm
import os
import matplotlib as mpl
from matplotlib import pyplot as plt

import EnsembleLib
from EnsembleLib import time_diff
import DatasetMaker

from collections import OrderedDict

import torch
from torch.optim import SGD, Adam, lr_scheduler
device = 'cuda'

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
tz = pytz.timezone('US/Eastern')
from scipy.stats import ortho_group

torch.set_default_dtype(torch.float64)

def getIsotropicSP(alpha, nu, omega, lam):
    x = alpha*(omega + 1) + nu*(lam - omega - 1)
    y = 4 * lam * nu**2 * (omega+1)
    q =  (np.sqrt(x**2+y)-x)/(2*nu)
    qhat = alpha/(lam+q)
    #qhat = (np.sqrt(x**2+y)+alpha*(1+omega)-nu*(lam+omega+1))/(2*lam*(1+omega))
    kappa = q+lam
    gamma = (4*alpha*nu*(1+omega)**2)/(np.sqrt(x**2+y) + alpha*(omega+1) + nu*(lam+omega+1))**2
    return q, qhat, gamma

def getIsotropicError_diag(alpha, nu, omega, lam, zeta, eta = 0, w=1, wbar=1):
    q, qhat, gamma = getIsotropicSP(alpha, nu, omega, lam)
    kappa = q+lam
    t = 1+qhat/nu*(1+omega)
    E = 1/(1-gamma)*((nu - qhat*(t+1)/t**2)*w**2 + (1-nu)*wbar**2) + gamma/(1-gamma)*zeta**2 + eta**2
    return E
    
#Helper function to invert RS Matrices quickly using sherman-morrison.
def invertRSMatrix(sigma):
    assert sigma.shape[0] == sigma.shape[1]
    n =sigma.shape[0]
    b = (torch.sum(sigma) - torch.sum(torch.diag(sigma)))/(n*(n-1))
    a = torch.mean(torch.diag(sigma))-b
    return 1/a*(torch.eye(n).to('cuda') - b/(a+n*b)*torch.ones(n,n).to('cuda'))
    
#Solve the saddle-point equations for a single linear readout
def solveLinearReadoutSP(sigma_s, sigma_0, A, alpha, lam, tol = 1e-6, max_iter = 2000, printEvery = 500, verbose = True, RS = False):
    
    N =A.shape[0]
    IN = torch.eye(N).to('cuda')
    M = sigma_s.shape[0]
    
    #Make sure dimensions are compatible.
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
    assert A.shape[1] == sigma_s.shape[0]
    
    nu = N/M
    
    sigma_tilde = 1/nu*A@(sigma_s+sigma_0)@(A.T)
    
    damp = 0
    diffdiff = 100*tol
    
    q = torch.tensor([0]).to('cuda')
    qhat = torch.tensor([0]).to('cuda')
    diff = 100
    
    for i in range(max_iter):
        qhat_new =damp*qhat + (1-damp)*alpha/(lam+q)
        
        if RS:
            CInv = invertRSMatrix(IN + qhat*sigma_tilde)
        else:
            CInv = torch.linalg.inv(IN + qhat*sigma_tilde)
        q_new = damp*q + (1-damp)*1/M*torch.trace(CInv@sigma_tilde)
        diff = torch.abs(qhat_new-qhat) + torch.abs(q_new-q)
        
        if diff<tol:
            
            gamma = alpha/M/(lam+q)**2*torch.trace(CInv@sigma_tilde@CInv@sigma_tilde)
            
            if verbose:
                print('Solved after ' + str(i) + ' iterations: diff = ' + str(diff) + '; q = ' + str(q_new.item()) + '; qhat = ' + str(qhat_new.item()))            
            return q, qhat, gamma

        if (i+1)%printEvery == 0 and verbose:
            print('Iter: ' + str(i) + '; diff = ' + str(diff.item()) +'; q = ' + str(q.item()) + '--' + str(q_new.item()) + '; qhat = ' + str(qhat.item()) + '--' +str(qhat_new.item()))

        if i>max_iter/10:
            damp = i/(2*max_iter)
        
        qhat = qhat_new
        q = q_new
    
    gamma = alpha/M/(lam+q)**2*torch.trace(CInv@sigma_tilde@CInv@sigma_tilde)
    return q, qhat, gamma

def calcDiagErrorFromSP(q, qhat, sigma_s, sigma_0, A, zeta, eta, w_star, lam, alpha):
    
    w_star = w_star.reshape((-1, 1))
    
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
    assert A.shape[1] == sigma_s.shape[0]
    assert w_star.shape[0] == sigma_s.shape[0]
    
    M = sigma_s.shape[0]
    N = A.shape[0]
    nu = N/M
    
    sigma_tilde = 1/nu*A@(sigma_s+sigma_0)@(A.T)
    G = torch.eye(N).to('cuda') + qhat*sigma_tilde
    GInv = torch.linalg.inv(G)
    
    kappa = lam+q
    gamma = alpha/(M*kappa*kappa)*torch.trace(GInv@sigma_tilde@GInv@sigma_tilde)
    
    error = 1/(1-gamma)/M*w_star.T@(sigma_s - 1/nu*qhat*sigma_s@A.T@GInv@A@sigma_s - 1/nu*qhat*sigma_s@A.T@GInv@GInv@A@sigma_s)@w_star + (gamma*zeta**2 + eta**2)/(1-gamma)
    return error

def calcOffDiagErrorFromSP(Q, Qhat, R, Rhat, sigma_s, sigma_0, Ar, Arp, zeta, w_star, lam, alpha):
    
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
    assert Ar.shape[1] == sigma_s.shape[0]
    assert Arp.shape[1] == sigma_s.shape[0]
    
    w_star = w_star.reshape((-1, 1))
    
    M = sigma_s.shape[0]
    Nr = Ar.shape[0]
    Nrp = Arp.shape[0]
    nur = Nr/M
    nurp = Nrp/M
    
    sigma_tilde_r= 1/nur*Ar@(sigma_s+sigma_0)@(Ar.T)
    sigma_tilde_rp= 1/nurp*Arp@(sigma_s+sigma_0)@(Arp.T)
    
    sigma_tilde_rrp = 1/((nur*nurp)**(1/2))*Ar@(sigma_s+sigma_0)@(Arp.T)
    
    Cr = torch.eye(Nr).to('cuda') + Qhat*sigma_tilde_r
    Crp = torch.eye(Nrp).to('cuda') + Rhat*sigma_tilde_rp
    
    CrInv = torch.linalg.inv(Cr)
    CrpInv = torch.linalg.inv(Crp)
    
    gamma = alpha/((lam+Q)*(lam+R))*1/M*torch.trace(CrInv@sigma_tilde_rrp@CrpInv@sigma_tilde_rrp.T)
    
    error = gamma/(1-gamma)*zeta**2 + 1/(1-gamma)/M*w_star.T@sigma_s@w_star -1/(1-gamma)/M*w_star.T@sigma_s@(Qhat/nur*Ar.T@CrInv@Ar + Rhat/nurp*Arp.T@CrpInv@Arp)@sigma_s@w_star + 1/(1-gamma)/M*Qhat*Rhat*1/((nur*nurp)**(1/2))*w_star.T@sigma_s@Ar.T@CrInv@sigma_tilde_rrp@CrpInv@Arp@sigma_s@w_star
    return error

#Wraps saddle point solver to get error curves for given A matrices
def getLinearErrorCurve(sigma_s, sigma_0, A_list, zeta, eta, w_star, lam, alphas, max_iter = 1000, tol = 1e-5, verbose = False, RS = False, UseGlobCorrSP=False):
    
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
    
    for A in A_list:
        assert A.shape[1] == sigma_s.shape[0]
    
    numAlphas = len(alphas)
    K = len(A_list)
    M = sigma_s.shape[0]
    
    #Order Parameters for each readout:
    qs = torch.empty((K, numAlphas), device = 'cuda')
    qhats = torch.empty((K, numAlphas), device = 'cuda')
    
    Couplings = torch.empty((K, K, numAlphas), device = 'cuda')#Errors of the individual readouts and coupling terms
    TotErrors = torch.empty(numAlphas, device = 'cuda')#Errors of the net predictors.
    
    for r, A in enumerate(A_list):
        A = A_list[r]
        for alphaInd in range(alphas.shape[0]):
            alpha = alphas[alphaInd]
            
            #For testing purposes, gives the option to use the simplified fixed point equations for global correlations
            if UseGlobCorrSP:
                q, qhat, _ = getGlobCorrSP(alpha, A.shape[0]/M, sigma_s[0,1].item(), lam)
            else:
                q, qhat, _ = solveLinearReadoutSP(sigma_s, sigma_0, A, alpha, lam,  tol, max_iter, printEvery = 50, verbose = verbose, RS = RS)
            qs[r, alphaInd] = q
            qhats[r, alphaInd] = qhat           
            Couplings[r, r, alphaInd] = calcDiagErrorFromSP(q, qhat, sigma_s, sigma_0, A, zeta, eta, w_star, lam, alpha)
            print('AlphaInd: ' + str(alphaInd))
        print('Finished Readout ' + str(r))
            
    for r in range(K):
        Ar = A_list[r]
        for rp in range(r+1, K):
            Arp = A_list[rp]
            for alphaInd in range(alphas.shape[0]):
                alpha = alphas[alphaInd]
                Q = qs[r, alphaInd]
                Qhat = qhats[r, alphaInd]
                R = qs[rp, alphaInd]
                Rhat = qhats[rp, alphaInd]
                
                error = calcOffDiagErrorFromSP(Q, Qhat, R, Rhat, sigma_s, sigma_0, Ar, Arp, zeta, w_star, lam, alpha)
                Couplings[r, rp, alphaInd] = error
                Couplings[rp, r, alphaInd] = error
                
    TotErrors = torch.mean(Couplings, axis = (0,1))
    return Couplings, TotErrors


#Theory of globally correlated data
def getGlobCorrSP(alpha, nu, c, lam, s = 1, omega = 0):
    
    #Order parameters diverge for infinite alpha
    assert alpha<np.inf
    
    a = s*(1-c) + omega
    
    x = alpha*a + nu*(lam-a)
    y = 4*lam*nu**2*a
    q =  (np.sqrt(x**2+y)-x)/(2*nu)
    qhat = alpha/(lam+q)
    gamma = alpha*nu*a**2/(lam+q)**2/(nu+qhat*a)**2
    return q, qhat, gamma

def getGlobCorrError_diag(alpha, nu, c, lam, zeta=0, eta = 0, s = 1, omega = 0, rho = 0):
    
    assert alpha>=0
    assert nu>0
    assert rho<=1 and rho>=-1
    
    a = s*(1-c)+omega
    F1 = (s*(1-c)*(1-nu)+omega)/nu
    omrs = 1-rho**2
    rs = rho**2
    
    #First treat the case of infinite data.
    #Note that in this case, the regularization does not matter, so we may simply use the ridgeless results at infinite alpha
    if alpha == np.inf:
        E0 = s*(1-c)*(1-s*(1-c)*nu/a) + eta**2
        E1 = F1 + eta**2
        return omrs*E0 + rs*E1
    
    #Next treat the ridgeless case:
    if lam == 0:
        if alpha<nu:
            E0 = s*nu*(1-c)/(nu-alpha)*(1 + s*alpha*(1-c)*(alpha-2*nu)/(nu*(a))) + (alpha*zeta**2 + nu*eta**2)/(nu-alpha)
            E1 = nu/(nu-alpha)*(F1) + (alpha*zeta**2 + nu*eta**2)/(nu-alpha)
            return omrs*E0 + rs*E1
        elif alpha>nu:
            E0 = s*alpha*(1-c)/(alpha-nu)*(1 - s*nu*(1-c)/(a)) + (nu*zeta**2 + alpha*eta**2)/(alpha-nu)
            E1 = alpha/(alpha-nu)*(F1) + (nu*zeta**2 + alpha*eta**2)/(alpha-nu)
            return omrs*E0 + rs*E1
        elif alpha==nu:
            return np.inf
    
    q, qhat, gamma = getGlobCorrSP(alpha, nu, c, lam)
    innerUp  = (1-c)*s*qhat*nu*(qhat*a+2*nu)
    innerDown = (qhat*a+nu)**2
    E0 = s*(1-c)/(1-gamma)*(1-innerUp/innerDown)+(gamma*zeta**2 + eta**2)/(1-gamma)
    E1 = 1/(1-gamma)*F1+(gamma*zeta**2 + eta**2)/(1-gamma)
    return omrs*E0 + rs*E1

def Calc_F_0(alpha, s, c, nu_r, nu_rrp, nu_rp, omega):
    
    #Without loss of generality, we may assume nu_r<= nu_rp
    nu_r, nu_rp = sorted([nu_r, nu_rp])
        
    if alpha <= nu_r  and alpha <= nu_rp:
        return (c-1) * s * (nu_r * nu_rp * ((2*alpha-1)*(c-1)*s + omega) - alpha**2 * (c-1) * s * nu_rrp) / (nu_r * ((c-1)*s - omega) * nu_rp)
    elif nu_r <=alpha and  alpha <= nu_rp:
        return (c-1) * s * (nu_rp * ((c-1)*s*nu_r + (alpha-1)*(c-1)*s + omega) - alpha * (c-1) * s * nu_rrp) / (((c-1)*s - omega) * nu_rp)
    elif nu_r <= alpha and nu_rp <= alpha:
        return (c-1) * s * ((c-1)*s*nu_rp - c*s*nu_rrp + (c-1)*s*nu_r - c*s + s*nu_rrp + s + omega) / ((c-1)*s - omega)
    else:
        raise ValueError('Invalid input values')

def getGlobCorrError_offDiag(alpha, nu_r, nu_rp, nu_rrp, c, lam, zeta, eta, s = 1, omega = 0, rho = 0):
    
    assert nu_rrp<nu_r<=1 and nu_rrp<nu_rp<=1
    
    #Without loss of generality, we may assume nu_r<= nu_rp
    nu_r, nu_rp = sorted([nu_r, nu_rp])
    
    omrs = 1-rho**2
    rs = rho**2
    a = s*(1-c) + omega
    
    #At infinite alpha, we may use the ridgeless result in the limit alpha -> Infinity
    if alpha == np.inf:
        E0 = (c-1) * s * ((c-1)*s*nu_rp - c*s*nu_rrp + (c-1)*s*nu_r - c*s + s*nu_rrp + s + omega) / ((c-1)*s - omega)
        E1 = (s*(1-c)*(nu_rrp-nu_r*nu_rp)+omega*nu_rrp)/(nu_r*nu_rp)
        return omrs*E0 + rs*E1
        
        # e1r =  nu_r*(1-c) + c
        # e2rp = e2rp = nu_rp*(1-c) + c
        # errp = c
        # return 1-e1r-e2rp+errp
    
    if lam==0:
        F_0 = Calc_F_0(alpha, s, c, nu_r, nu_rrp, nu_rp, omega)
        F_1 = (s*(1-c)*(nu_rrp-nu_r*nu_rp)+omega*nu_rrp)/(nu_r*nu_rp)
        gamma = 4*alpha*nu_rrp/(alpha+nu_r+abs(alpha-nu_r))/(alpha+nu_rp+abs(alpha-nu_rp))
        E0 = 1/(1-gamma)*F_0 + gamma/(1-gamma)*zeta**2
        E1 = 1/(1-gamma)*F_1 + gamma/(1-gamma)*zeta**2
        return omrs*E0 + rs*E1
        
#         if alpha<nu_r:
#             e1r = alpha*(1-c) + c
#         else:
#             e1r = nu_r*(1-c) + c
        
#         if alpha<nu_rp:
#             e2rp = alpha*(1-c) + c
#         else:
#             e2rp = nu_rp*(1-c) + c
        
#         errp = c
#         return 1-e1r-e2rp+errp
        
    q_r, qhat_r, _ = getGlobCorrSP(alpha, nu_r, c, lam, s, omega)
    q_rp, qhat_rp, _ = getGlobCorrSP(alpha, nu_rp, c, lam, s, omega)
    
    gamma = a**2*alpha*nu_rrp/((lam+q_r)*(lam+q_rp)*(a*qhat_r+nu_r)*(a*qhat_rp+nu_rp))
    sub_r  = s*(1-c)*nu_r*qhat_r/(nu_r+a*qhat_r)
    sub_rp  = s*(1-c)*nu_rp*qhat_rp/(nu_rp+a*qhat_rp)
    sub_rrp = s*(1-c)*a*nu_rrp*qhat_r*qhat_rp/(nu_r+a*qhat_r)/(nu_rp+a*qhat_rp)
    E0 = s*(1-c)/(1-gamma)*(1-sub_r-sub_rp+sub_rrp) + gamma/(1-gamma)*zeta**2
    E1 = 1/(1-gamma)*((s*(1-c)*(nu_rrp-nu_r*nu_rp)+omega*nu_rrp)/(nu_r*nu_rp)) + gamma/(1-gamma)*zeta**2
    return omrs*E0 + rs*E1
#     e1r = nu_r*(1-c)**2*Qhat/(nu_r+(1-c)*Qhat) + c
#     e2rp = nu_rp*(1-c)**2*Rhat/(nu_rp+(1-c)*Rhat) + c
    # errp = c
    # error = 1-e1r-e2rp+errp

#assumes nus is a vector of subsampling fractions.  
def getGlobCorrError_exclusive(alpha, nus, c, lam=0, zeta=0, eta=0, s = 1, omega = 0, rho = 0):
    count = 0
    errors = np.zeros((len(nus), len(nus)))
    for r in range(len(nus)):
        nu = nus[r]
        if nu>0:
            errors[r,r]=(getGlobCorrError_diag(alpha, nu, c, lam, zeta, eta, s, omega, rho))
            count+=1
            for rp in range(r+1, len(nus)):
                nup = nus[rp]
                if nup>0:
                    count+=2
                    #No overlaps: set nurrp=0
                    curErr = getGlobCorrError_offDiag(alpha, nu, nup, 0, c, lam, zeta, eta, s, omega, rho)
                    errors[r, rp] = curErr
                    errors[rp, r] = curErr
    return np.sum(errors)/count

#assumes nus is a matrix of subsampling fractions (diagonal) and overlaps (off-diagonal).  
def getGlobCorrError(alpha, nus, c, lam=0, zeta=0, eta=0, s = 1, omega = 0, rho = 0, exclusive = False):
    
    if exclusive:
        return getGlobCorrError_exclusive(alpha, nus, c, lam, zeta, eta, s, omega, rho)
    
    count = 0
    l, temp = nus.shape
    assert l == temp #nus must be square
    
    errors = np.zeros((len(nus), len(nus)))
    for r in range(len(nus)):
        nu = nus[r, r]
        if nu>0:
            errors[r,r]=(getGlobCorrError_diag(alpha, nu, c, lam, zeta, eta, s, omega, rho))
            count+=1
            for rp in range(r+1, len(nus)):
                nup = nus[rp, rp]
                if nup>0:
                    nurrp = nus[r, rp]
                    count+=2
                    curErr =getGlobCorrError_offDiag(alpha, nu, nup, nurrp, c, lam, zeta, eta, s, omega, rho)
                    errors[r, rp] = curErr
                    errors[rp, r] = curErr
    return np.sum(errors)/count

#Currently assumes no overlaps
def getGlobCorrError_Homog_Exclusive(alpha, k, c, nu_0 = 0, lam=0, zeta=0, eta=0, s = 1, omega = 0, rho = 0):
    if k == np.inf:
        #We know that in this limit alpha>nu = 1/k because nu -> 0
        return s*(1-c)*(1-rho**2)+rho**2*omega
    
    nu = nu_0/k
    diag_err = getGlobCorrError_diag(alpha, nu, c, lam, zeta, eta, s, omega, rho)
    off_diag_err = getGlobCorrError_offDiag(alpha, nu, nu, 0, c, lam, zeta, eta, s, omega, rho)
    tot_err = 1/k*(diag_err + (k-1)*off_diag_err)
    return tot_err

def getGlobCorrErrorCurve(alphas, nus, c, lam = 0, zeta = 0, eta = 0, s = 1, omega = 0, rho = 0, exclusive = False):
    if exclusive:
        return np.array([getGlobCorrError_exclusive(alpha, nus, c, lam, zeta, eta, s = s, omega = omega, rho = rho) for alpha in alphas])
    return np.array([getGlobCorrError(alpha, nus, c, lam, zeta, eta, s = s, omega = omega, rho = rho) for alpha in alphas])

#Homogeneous exclusive errror curves.
def getGlobCorrErrorCurve_Homog_Exclusive(alphas, k, c, nu_0 = 1, lam = 0, zeta = 0, eta = 0, s = 1, omega = 0, rho = 0):
    nu = nu_0/k
    return np.array([getGlobCorrError_Homog_Exclusive(alpha, k, c, nu_0, lam, zeta, eta, s = s, omega = omega, rho = rho) for alpha in alphas])