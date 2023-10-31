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
def getLinearErrorCurve(sigma_s, sigma_0, A_list, zeta, eta, w_star, lam, alphas, max_iter = 1000, tol = 1e-5, verbose = False, RS = False, UseEquiCorrSP=False):
    
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
            
            #For testing purposes, gives the option to use the simplified fixed point equations for equicorrelated data
            if UseEquiCorrSP:
                q, qhat, _ = getEquiCorrSP(alpha, A.shape[0]/M, sigma_s[0,1].item(), lam)
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


#Theory of equicorrelated data
#Updating to handle negative regularization
def getEquiCorrSP(alpha, nu, c, lam, s = 1, omega = 0):
    
    #Order parameters diverge for infinite alpha
    assert alpha<np.inf
    assert lam!=0
    
    a = s*(1-c) + omega**2
    
    #Handle zero-data limit
    if alpha == 0:
        q = a
        qhat = 0
        gamma = 0
        return q, qhat, gamma
    
    x = alpha*a + nu*(lam-a)
    y = 4*lam*nu**2*a
    q =  np.max([(np.sqrt(x**2+y)-x)/(2*nu), (-np.sqrt(x**2+y)-x)/(2*nu)]) #Maximum will select positive value whether lambda is positive or negative.
    qhat = alpha/(lam+q)
    S = qhat/(nu+a*qhat)
    gamma = nu* a**2 * S**2 / alpha
    return q, qhat, gamma

def getEquiCorrError_diag(alpha, nu, c, lam, zeta=0, eta = 0, s = 1, omega = 0, rho = 0):
    
    assert alpha>=0
    assert nu>0
    assert rho<=1 and rho>=-1

    a = s*(1-c) + omega**2
    
    #At infinite alpha, we may use the ridgeless result in the limit alpha -> Infinity
    if alpha == np.inf:
        S = 1/a
        gamma = 0
    #Treat Zero Regularization (ridgeless limit)
    elif alpha == 0:
        S = 0
        gamma = 0
    elif lam==0:
        #Special case where gamma = 1 so we must get rid of numerical instability.  This limit is shown in EquiCorrOptimalRegularization.nb
        if alpha == 1 and nu==1 and omega == 0 and zeta == 0 and eta == 0:
            return 0
        S = 2*alpha/a/(alpha+nu + np.abs(alpha-nu))
        gamma = 4*alpha*nu/(alpha+nu+np.abs(alpha-nu))**2
    #Now Treat General Case.
    else:
        q, qhat, gamma = getEquiCorrSP(alpha, nu, c, lam, s, omega)
        S = qhat/(nu+a*qhat)
        
    #Set the values of I0 and I1
    I0 = s*(1-c)*(1 - 2*s*(1-c)*nu*S + a*s*(1-c)*nu*S*S)
    
    if c == 0:
        I1 = I0
    else:
        I1 = (s*(1-c)*nu*(1-nu)+omega**2*nu)/(nu*nu)
    
    return 1/(1-gamma)*((1-rho**2)*I0 + rho**2*I1 + gamma*zeta**2 + eta**2)

#Note: Off-diagonal terms don't depend on the readout noise.
def getEquiCorrError_offDiag(alpha, nu_r, nu_rp, nu_rrp, c, lam, zeta, s = 1, omega = 0, rho = 0):
    
    assert nu_rrp<nu_r<=1 and nu_rrp<nu_rp<=1
    
    #Without loss of generality, we may assume nu_r<= nu_rp
    nu_r, nu_rp = sorted([nu_r, nu_rp])
    
    a = s*(1-c) + omega**2
    
    #At infinite alpha, we may use the ridgeless result in the limit alpha -> Infinity
    if alpha == np.inf:
        Sr = 1/a
        Srp = 1/a
        gamma = 0
    #Treat Zero Regularization (ridgeless limit)
    elif alpha==0:
        Sr = 0
        Srp = 0
        gamma = 0
    elif lam==0:
        Sr = 2*alpha/a/(alpha+nu_r + np.abs(alpha-nu_r))
        Srp = 2*alpha/a/(alpha+nu_rp + np.abs(alpha-nu_rp))
        gamma = 4*alpha*nu_rrp/(alpha+nu_r+np.abs(alpha-nu_r))/(alpha+nu_rp+np.abs(alpha-nu_rp))
    #Now Treat General Case.
    else:
        q_r, qhat_r, _ = getEquiCorrSP(alpha, nu_r, c, lam, s, omega)
        q_rp, qhat_rp, _ = getEquiCorrSP(alpha, nu_rp, c, lam, s, omega)
        Sr = qhat_r/(nu_r+a*qhat_r)
        Srp = qhat_rp/(nu_rp+a*qhat_rp)
        gamma = a**2*nu_rrp*Sr*Srp/alpha
        
    #Set the values of I0 and I1
    I0 = s*(1-c)*(1-s*(1-c)*nu_r*Sr - s*(1-c)*nu_rp*Srp + a*s*(1-c)*nu_rrp*Sr*Srp)
    
    if c == 0:
        I1 = I0
    else:
        I1 = (s*(1-c)*(nu_rrp-nu_r*nu_rp)+omega**2*nu_rrp)/(nu_r*nu_rp)
    
    return 1/(1-gamma)*((1-rho**2)*I0 + rho**2*I1 + gamma*zeta**2)

#assumes nus is a vector of subsampling fractions.  
def getEquiCorrError_exclusive(alpha, nus, c, lam=0, zeta=0, eta=0, s = 1, omega = 0, rho = 0):
    count = 0
    errors = np.zeros((len(nus), len(nus)))
    for r in range(len(nus)):
        nu = nus[r]
        if nu>0:
            errors[r,r]=(getEquiCorrError_diag(alpha, nu, c, lam, zeta, eta, s, omega, rho))
            count+=1
            for rp in range(r+1, len(nus)):
                nup = nus[rp]
                if nup>0:
                    count+=2
                    #No overlaps: set nurrp=0
                    curErr = getEquiCorrError_offDiag(alpha, nu, nup, 0, c, lam, zeta, s, omega, rho)
                    errors[r, rp] = curErr
                    errors[rp, r] = curErr
    return np.sum(errors)/count

#assumes nus is a matrix of subsampling fractions (diagonal) and overlaps (off-diagonal).  
def getEquiCorrError(alpha, nus, c, lam=0, zeta=0, eta=0, s = 1, omega = 0, rho = 0, exclusive = False):
    
    if exclusive:
        return getEquiCorrError_exclusive(alpha, nus, c, lam, zeta, eta, s, omega, rho)
    
    count = 0
    l, temp = nus.shape
    assert l == temp #nus must be square
    
    errors = np.zeros((len(nus), len(nus)))
    for r in range(len(nus)):
        nu = nus[r, r]
        if nu>0:
            errors[r,r]=(getEquiCorrError_diag(alpha, nu, c, lam, zeta, eta, s, omega, rho))
            count+=1
            for rp in range(r+1, len(nus)):
                nup = nus[rp, rp]
                if nup>0:
                    nurrp = nus[r, rp]
                    count+=2
                    curErr =getEquiCorrError_offDiag(alpha, nu, nup, nurrp, c, lam, zeta, s, omega, rho)
                    errors[r, rp] = curErr
                    errors[rp, r] = curErr
    return np.sum(errors)/count

#Currently assumes no overlaps
def getEquiCorrError_Homog_Exclusive(alpha, k, c, nu_0 = 1, lam=0, zeta=0, eta=0, s = 1, omega = 0, rho = 0):
    
    if k == np.inf:
        #We know that in this limit alpha>nu = 1/k because nu -> 0
        return (s*(1-c)*(1-rho**2)) + (rho**2 * omega**2)
    
    nu = nu_0/float(k)
    
    if lam == 'local':
        lam = getEquiCorr_OptReg_Local(zeta, eta, nu, c, s, rho, omega)
    elif lam == 'global':
        lam = getEquiCorr_OptReg_Ens_HomogExclusive(k, zeta, eta, nu, c, s, rho, omega, alpha)
    else:
        #print(f'lambda = {lam}')
        #assert isinstance(lam, (int, float)), "Regularization must be a number or 'local' or 'global'"
        lam = float(lam)
    
    diag_err = getEquiCorrError_diag(alpha, nu, c, lam, zeta, eta, s, omega, rho)
    off_diag_err = getEquiCorrError_offDiag(alpha, nu, nu, 0, c, lam, zeta, s, omega, rho)
    tot_err = 1/k*(diag_err + (k-1)*off_diag_err)
    return tot_err
                          
def getEquiCorrErrorCurve(alphas, nus, c, lam = 0, zeta = 0, eta = 0, s = 1, omega = 0, rho = 0, exclusive = False):
    
    errorCurve = np.zeros(len(alphas))
    
    if exclusive:
        for alphaInd, alpha in enumerate(alphas):
            errorCurve[alphaInd] = getEquiCorrError_exclusive(alpha, nus, c, lam, zeta, eta, s = s, omega = omega, rho = rho)
    else:
        for alphaInd, alpha in enumerate(alphas):
            errorCurve[alphaInd] = getEquiCorrError(alpha, nus, c, lam, zeta, eta, s = s, omega = omega, rho = rho)
    
    return errorCurve

#Homogeneous exclusive errror curves.
def getEquiCorrErrorCurve_Homog_Exclusive(alphas, k, c, nu_0 = 1, lam = 0, zeta = 0, eta = 0, s = 1, omega = 0, rho = 0):
    
    if k==np.inf:
        nu = 0
    else:
        nu = np.divide(nu_0, float(k))
    
    if lam == 'local': #Indicates locally optimal regularization, which is constant with alpha
        lam = getEquiCorr_OptReg_Local(zeta, eta, nu, c, s, rho, omega)
        
    # errorCurve = np.zeros(len(alphas))
    # for alphaInd, alpha in enumerate(alphas):
    #     errorCurve[alphaInd] = getEquiCorrError_Homog_Exclusive(alpha, k, c, nu_0, lam, zeta, eta, s = s, omega = omega, rho = rho)
    #return errorCurve    
    
    return np.vectorize(getEquiCorrError_Homog_Exclusive, excluded=['k', 'c', 'nu_0', 'lam', 'zeta', 'eta', 's', 'omega', 'rho'])(alphas, k, c, nu_0, lam, zeta, eta, s = s, omega = omega, rho = rho)

def getEquiCorr_OptReg_Local(zeta, eta, nu, c, s, rho, omega):
    a=s*(1-c) + omega**2
    numerator = (zeta**2 + eta**2) * nu + (-1 + c) * s * (-nu + (-1 + 2 * nu) * rho**2) + rho**2 * omega**2
    denominator = (-1 + c)**2 * s**2 * nu**2 * (-1 + rho**2)
    return a * (-1 - a * numerator / denominator)

def getEquiCorr_OptReg_Ens_HomogExclusive(k, zeta, eta, nu, c, s, rho, omega, alpha):
    
    # Calculate the constant 'a' based on the provided relation
    a_const = s * (1 - c) + omega ** 2

    # Compute the coefficient of the S^4 term for the quartic polynomial
    a_4 = a_const ** 4 * (-1 + c) ** 2 * (-1 + k) * s ** 2 * nu ** 3 * (-1 + rho ** 2)
    
    # The S^3 term is not present in the polynomial, so its coefficient is 0
    a_3 = 0  
    
    # Compute the coefficient of the S^2 term
    a_2 = -a_const ** 2 * (-1 + c) ** 2 * (-3 + 2 * k) * s ** 2 * alpha * nu ** 2 * (-1 + rho ** 2)
    
    # Compute the coefficient of the S^1 term
    a_1 = (-a_const * (-1 + c) ** 2 * s ** 2 * alpha ** 2 * nu * (-1 + rho ** 2)) + a_const ** 2 * alpha * (zeta ** 2 * nu + eta ** 2 * nu + (-1 + c) * s * (-rho ** 2 + nu * (-1 + 2 * rho ** 2)) +  rho ** 2 * omega ** 2)
    
    # Compute the constant term (S^0 coefficient) of the polynomial
    a_0 = (-1 + c) ** 2 * k * s ** 2 * alpha ** 2 * nu * (-1 + rho ** 2)

    # Formulate the quartic polynomial with the computed coefficients
    coefficients = [a_4, a_3, a_2, a_1, a_0]
    # Use numpy to find the roots of the quartic polynomial
    roots = np.roots(coefficients)
    #We find that the last value in this array corresponds to the physical solution for S
    
    if roots.size == 0:
        return 0
    
    # If roots has multiple values, return the last one
    S = roots[-1]

    lam = (-1 + a_const * S) * (-alpha + a_const * S * nu) / (S * nu)
    
    return lam