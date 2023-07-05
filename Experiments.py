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
import auxFuncs

from collections import OrderedDict

import torch
from torch.optim import SGD, Adam, lr_scheduler
device = 'cuda'

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
tz = pytz.timezone('US/Eastern')
from scipy.stats import ortho_group

def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def convert_to_list(variable):
    if isinstance(variable, list):
        return variable
    else:
        return [variable]

def matrix_sqrt(A):
    L,V = torch.linalg.eigh(A)
    DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.T

def runEvP_Class_Subsamp_Stringer(expt_name, f, eta_r_scale, projType, learningRule, num_samples_list, nVals, lamVals, zeta = 0, num_trials = 5):
    
    dat = np.load(f, allow_pickle=True).item()
    x, y = DatasetMaker.makeDenseDisc_Dataset(dat)#, zscore = True)
    M = x.shape[1]
    sigma_0 = 0*torch.eye(M)
    runEvP_Class_Subsamp(expt_name, x,y, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, nVals, lamVals, zeta, num_trials)
    
def runEvP_Class_Subsamp_Gauss(expt_name, w_star, sigma_s, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, NuVals, lamVals, zeta = 0, num_trials = 5):
    M = w_star.shape[0]
    N = 50000
    x, y = DatasetMaker.makeGaussianDataset_sgn(N, w_star, sigma_s, zeta)
    nVals = [int(M*nu) for nu in NuVals]
    runEvP_Class_Subsamp(expt_name, x, y, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, nVals, lamVals, zeta, num_trials)

def runEvP_Class_Subsamp(expt_name, x, y, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, nVals, lamVals, zeta, num_trials):
    OutStrategy = 'Maj' #Stands for majority vote.  In this case, there is only one vote to count.
    M = x.shape[1]
      
    exptPath = expt_name
    print(expt_name)
    if not os.path.exists(exptPath):
        # Create a new directory because it does not exist 
        os.makedirs(exptPath)
        print("The new experiment directory is created!")
    else:
        print("The experiment directory already exists!")
    
    numNVals = len(nVals)
    numLamVals = len(lamVals)
    numPVals = len(num_samples_list)
    
    if projType == 'rp':
        align_proj = False
    elif projType == 'np':
        align_proj = True
    else:
        raise Exception('Invalid projection type. Must be np or rp')
    
#   tr_loss = np.empty([numPVals, numKVals, num_trials])
#   test_loss = np.empty([numPVals, numKVals, num_trials])
#   tr_acc = np.empty([numPVals, numKVals, num_trials])
#   test_acc = np.empty([numPVals, numKVals, num_trials])
    
    tr_err = np.empty([numPVals, numNVals, numLamVals, num_trials])
    test_err = np.empty([numPVals, numNVals, numLamVals, num_trials])

    for PInd,P in enumerate(num_samples_list):
        for nInd, n in enumerate(nVals):
            start_time = datetime.now(tz)
            eta_rs = [eta_r_scale]
            
            ns = [int(n)]
            student = EnsembleLib.MultiReadout(M, ns, sigma_0, eta_rs, auxFuncs.sign, auxFuncs.majorityVote, align_proj = align_proj)
            
            for lamInd, lam in enumerate(lamVals):
                for trial in range(num_trials):
                    train_loader, test_loader = DatasetMaker.split_data(P, x, y)

                    if learningRule == 'lr':
                        student.regress_readout(train_loader, test_loader, lam)
                    else:
                        raise Exception('Only lr is supported at the moment.')
                    
                    errors = student.eval(train_loader, test_loader, auxFuncs.SgnErrorRate)
                    tr_err[PInd, nInd, lamInd, trial] = errors[0]
                    test_err[PInd, nInd, lamInd, trial] = errors[1]
                
                print('P ='+str(P)+ '; n = ' + str(n) + '; lam = ' + str(lam) + '; Eg = ' + str(np.mean(test_err[PInd, nInd,:])) + '; time: ' + str(time_diff(start_time, datetime.now(tz))))
    
    outputDict = {}
    
    outputDict['num_samples_list'] = num_samples_list
    outputDict['n_list'] = nVals
    outputDict['lam_list'] = lamVals
    outputDict['tr_err'] = tr_err
    outputDict['test_err'] = test_err
    outputDict['Ordering'] = ['PInd', 'nInd', 'lamInd', 'trial']#The order of the indices in the tr_err and test_err tensors
    outputDict['projType'] = 'np'
    outputDict['learningRule'] = 'lr'
    outputDict['outputStrategy'] = 'Majority Vote'
    np.save(exptPath+'/output_dict', outputDict)
    
#Learning Curves for Different Distriutons of Connectivity.
def runEvP_Class_Ensemble_CompDists(expt_name, x, y, ConnDists, ensFunc, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, KVals, lamVals, num_trials, shuffle_degrees):
    
    if learningRule != 'lr':
        raise Exception('Only lr is supported at the moment.')
    
    M = x.shape[1]
    
    exptPath = expt_name
    print(expt_name)
    if not os.path.exists(exptPath):
        # Create a new directory because it does not exist 
        os.makedirs(exptPath)
        print("The new experiment directory is created!")
    else:
        print("The experiment directory already exists!")
        
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    numPVals = len(num_samples_list)
    
    if projType == 'rp':
        align_proj = False
    elif projType == 'np':
        align_proj = True
    else:
        raise Exception('Invalid projection type. Must be np or rp')
    
#   tr_loss = np.empty([numPVals, numKVals, num_trials])
#   test_loss = np.empty([numPVals, numKVals, num_trials])
#   tr_acc = np.empty([numPVals, numKVals, num_trials])
#   test_acc = np.empty([numPVals, numKVals, num_trials])
    
    tr_err = np.empty([numKVals, len(ConnDists), numPVals, numLamVals, num_trials])
    test_err = np.empty([numKVals, len(ConnDists), numPVals, numLamVals, num_trials])

    ns = EnsembleLib.genNs(KVals[0], M, ConnDists[0])
    student = EnsembleLib.MultiReadout(M, ns, sigma_0, eta_r_scale, auxFuncs.sign, align_proj = align_proj)
    
    for KInd, K in enumerate(KVals):
        for ConnDistInd, ConnDist in enumerate(ConnDists):
            ns = EnsembleLib.genNs(K, M, ConnDist)
            print(ns)
            student.setDegrees(ns)
            for PInd, P in enumerate(num_samples_list):
                start_time = datetime.now(tz)
                for trial in range(num_trials):
                    if shuffle_degrees:
                        ns = EnsembleLib.genNs(K, M, ConnDist)
                        student.setDegrees(ns)

                    train_loader, test_loader = DatasetMaker.split_data(P, x, y)
                    
                    for lamInd, lam in enumerate(lamVals):
                        student.regress_readout(train_loader, test_loader, lam)
                        errors = student.eval(train_loader, test_loader, ensFunc, auxFuncs.SgnErrorRate)
                        tr_err[KInd, ConnDistInd, PInd, lamInd, trial] = errors[0]
                        test_err[KInd, ConnDistInd, PInd, lamInd, trial] = errors[1]
                            
                print(f'K = {K}; ConnDist = {ConnDist}; P = {P}; Eg = {np.mean(test_err[KInd, ConnDistInd, PInd, :, :])}; time: {time_diff(start_time, datetime.now(tz))}')
                # print('K ='+str(K)+ '; ConnDist = ' + ConnDist + '; P = ' + str(P) +  '; Eg = ' + str(np.mean(test_err[KInd, ConnDistInd, PInd, :, :, :])) + '; time: ' + str(time_diff(start_time, datetime.now(tz))))
            
    outputDict = {}
    outputDict['num_samples_list'] = num_samples_list
    outputDict['lam_list'] = lamVals
    outputDict['ConnDists'] = ConnDists
    outputDict['KVals'] = KVals
    outputDict['tr_err'] = tr_err
    outputDict['test_err'] = test_err
    outputDict['Ordering'] = ['KInd', 'ConnDistInd', 'PInd', 'lamInd', 'trial']#The order of the indices in the tr_err and test_err tensors
    outputDict['projType'] = 'np'
    outputDict['learningRule'] = 'lr'
    np.save(exptPath+'/output_dict', outputDict)
    
#Compare Score-Averaging to Majority Vote for Different Distriutons of Connectivity.
def EnsFuncComparison(expt_name, x, y, ConnDists, ensFuncs, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, KVals, lamVals, num_trials, shuffle_degrees):
    
    if learningRule != 'lr':
        raise Exception('Only lr is supported at the moment.')
    
    M = x.shape[1]
    sigma_0 = 0*torch.eye(M)
    
    exptPath = expt_name
    print(expt_name)
    if not os.path.exists(exptPath):
        # Create a new directory because it does not exist 
        os.makedirs(exptPath)
        print("The new experiment directory is created!")
    else:
        print("The experiment directory already exists!")
        
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    numPVals = len(num_samples_list)
    
    if projType == 'rp':
        align_proj = False
    elif projType == 'np':
        align_proj = True
    else:
        raise Exception('Invalid projection type. Must be np or rp')
    
#   tr_loss = np.empty([numPVals, numKVals, num_trials])
#   test_loss = np.empty([numPVals, numKVals, num_trials])
#   tr_acc = np.empty([numPVals, numKVals, num_trials])
#   test_acc = np.empty([numPVals, numKVals, num_trials])
    
    tr_err = np.empty([numKVals, len(ConnDists), numPVals, numLamVals, len(ensFuncs), num_trials])
    test_err = np.empty([numKVals, len(ConnDists), numPVals, numLamVals, len(ensFuncs), num_trials])

    ns = EnsembleLib.genNs(KVals[0], M, ConnDists[0])
    student = EnsembleLib.MultiReadout(M, ns, sigma_0, eta_r_scale, auxFuncs.sign, align_proj = align_proj)
    
    
    
    for KInd, K in enumerate(KVals):
        for ConnDistInd, ConnDist in enumerate(ConnDists):
            ns = EnsembleLib.genNs(K, M, ConnDist)
            print(ns)
            student.setDegrees(ns)
            for PInd, P in enumerate(num_samples_list):
                start_time = datetime.now(tz)
                for trial in range(num_trials):
                    if shuffle_degrees:
                        ns = EnsembleLib.genNs(K, M, ConnDist)
                        student.setDegrees(ns)

                    train_loader, test_loader = DatasetMaker.split_data(P, x, y)
                    
                    for lamInd, lam in enumerate(lamVals):
                        
                        student.regress_readout(train_loader, test_loader, lam)
                        
                        for ensFuncInd, ensFunc in enumerate(ensFuncs):
                            errors = student.eval(train_loader, test_loader, ensFunc, auxFuncs.SgnErrorRate)
                            tr_err[KInd, ConnDistInd, PInd, lamInd, ensFuncInd, trial] = errors[0]
                            test_err[KInd, ConnDistInd, PInd, lamInd, ensFuncInd, trial] = errors[1]
                            

                print('K ='+str(K)+ '; ConnDist = ' + ConnDist + '; P = ' + str(P) +  '; Eg = ' + str(np.mean(test_err[KInd, ConnDistInd, PInd, :, :, :])) + '; time: ' + str(time_diff(start_time, datetime.now(tz))))
            
    outputDict = {}
    outputDict['num_samples_list'] = num_samples_list
    outputDict['lam_list'] = lamVals
    outputDict['ConnDists'] = ConnDists
    outputDict['KVals'] = KVals
    outputDict['EnsFuncs'] = ensFuncs
    outputDict['tr_err'] = tr_err
    outputDict['test_err'] = test_err
    outputDict['Ordering'] = ['KInd', 'ConnDistInd', 'PInd', 'lamInd', 'ensFuncInd', 'trial']#The order of the indices in the tr_err and test_err tensors
    outputDict['projType'] = 'np'
    outputDict['learningRule'] = 'lr'
    np.save(exptPath+'/output_dict', outputDict)
    
# def EnsFuncComparison_Gauss(expt_name, w_star, sigma_s, zeta, ConnDists, ensFuncs, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, KVals, lamVals, num_trials, shuffle_degrees):
#     M = w_star.shape[0]
#     N = 50000
#     x, y = DatasetMaker.makeGaussianDataset_sgn(N, w_star, sigma_s, zeta)
#     print(x[:10])
#     print(y[:10])
#     EnsFuncComparison(expt_name, x, y, ConnDists, ensFuncs, sigma_0, eta_r_scale, projType, learningRule, num_samples_list, KVals, lamVals, num_trials, shuffle_degrees)

#Simulate linear regression given a set of deterministic masks, feature noise covariance, etc.
#Option to vary over regularizations.
#Option to average over ground truth weights w
def runEvP_LinFixedMasks(expt_name, sigma_s, sigma_0, ns, eta, zeta, lams, num_samples_list, num_trials, exclusive = False, aveGTW = False, rho = -2):
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
    rho_init = rho
    for n in ns:
        assert n<=sigma_s.shape[0]
        
    lams = convert_to_list(lams)
    numLamVals = len(lams)
    
    M = sigma_s.shape[0]
    numPVals = len(num_samples_list)
    
    #Deal with rho.  If rho is used, set wstar accordingly.  If it is not used (default -2) set it to zero and don't project
    project = True
    if rho<-1:
        project=False
        rho = 0
        
    exptPath = expt_name
    print(expt_name)
    if not os.path.exists(exptPath):
        # Create a new directory because it does not exist 
        os.makedirs(exptPath)
        print("The new experiment directory is created!")
    else:
        print("The experiment directory already exists!")
        
    tr_err = np.empty([numPVals, numLamVals, num_trials])
    test_err = np.empty([numPVals, numLamVals, num_trials])
    
    #Starts with one big dataset and a single w_star    
    w_star =  torch.normal(torch.zeros(M), torch.ones(M)).to('cuda')#Randomly chosen w_star.
    if project:
        w_star = ((1-rho**2)**(1/2))*(w_star - torch.mean(w_star)) + rho
    D = int(2*np.max(num_samples_list[-1]))
    x, y = DatasetMaker.makeGaussianDataset_lin(int(D), w_star, sigma_s, zeta)
    
    student = EnsembleLib.MultiReadout(M, ns, sigma_0, eta, auxFuncs.identity, align_proj = True, exclusive = exclusive)
    A_list = student.A_list
    
    for PInd, P in enumerate(num_samples_list):
        start_time = datetime.now(tz)
        for lamInd, lam in enumerate(lams): 
            for trial in range(num_trials): 
                #Draw a new weight for each trial if desired
                if aveGTW:
                    w_star =  torch.normal(torch.zeros(M), torch.ones(M)).to('cuda')#Randomly chosen w_star.
                    if project:
                        w_star = ((1-rho**2)**(1/2))*(w_star - torch.mean(w_star)) + rho
                    x, y = DatasetMaker.makeGaussianDataset_lin(int(P), w_star, sigma_s, zeta)
                train_loader, test_loader = DatasetMaker.split_data(P, x, y)
                student.regress_readout(train_loader, test_loader, lam)
                errors = student.eval(train_loader, test_loader, auxFuncs.mean, auxFuncs.SquareError)
                #tr_err[PInd, lamInd, trial] = errors[0]
                #test_err[PInd, lamInd, trial] = errors[1]
                test_err[PInd, lamInd, trial] = student.calcEg_linear(sigma_s, w_star)
        
        print('P = ' + str(P) +  'Eg = ' + str(np.mean(test_err[PInd, :])) + '; time: ' + str(time_diff(start_time, datetime.now(tz))))
           
    #Store results and everything needed for theory plots in output dict.        
    outputDict = {}
    outputDict['num_samples_list'] = num_samples_list
    outputDict['lams'] = lams
    outputDict['A_list'] = A_list
    outputDict['tr_err'] = tr_err
    outputDict['test_err'] = test_err
    outputDict['Ordering'] = ['PInd', 'lamInd', 'trial']#The order of the indices in the tr_err and test_err tensors
    outputDict['projType'] = 'np'
    outputDict['sigma_s'] = sigma_s
    outputDict['sigma_0'] = sigma_0
    outputDict['eta'] = eta
    outputDict['zeta'] = zeta
    outputDict['aveGTW'] = aveGTW
    outputDict['rho'] = rho_init
    if aveGTW == False:
        outputDict['w_star'] = w_star
    np.save(exptPath+'/output_dict', outputDict)
    
#Simulate linear regression given a set of deterministic masks, feature noise covariance, etc.
def runEvP_Lin_CompDists(expt_name, sigma_s, sigma_0, connDists, KVals, eta, zeta, lams, num_samples_list, num_trials, shuffle_degrees = True):
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
        
    M = sigma_s.shape[0]
    numPVals = len(num_samples_list)
    numKVals = len(KVals)
    numLamVals = len(lams)
    numDists = len(connDists)
    
    w_star =  torch.normal(torch.zeros(M), torch.ones(M)).to('cuda')
    
    D = 10000
    x, y = DatasetMaker.makeGaussianDataset_lin(D, w_star, sigma_s, zeta)
    
    exptPath = expt_name
    print(expt_name)
    if not os.path.exists(exptPath):
        # Create a new directory because it does not exist 
        os.makedirs(exptPath)
        print("The new experiment directory is created!")
    else:
        print("The experiment directory already exists!")
        
    #tr_err = np.empty([numDists, numKVals, numLamVals, numPVals, num_trials])
    test_err = np.empty([numDists, numKVals, numLamVals, numPVals, num_trials])
    
    ns = EnsembleLib.genNs(1, M) #will be replaced before running anything.
    
    student = EnsembleLib.MultiReadout(M, ns, sigma_0, eta, auxFuncs.identity, align_proj = True)
     
    for connDistInd, connDist in enumerate(connDists):
        for kInd, k in enumerate(KVals):
            ns = EnsembleLib.genNs(k, M, connDist)
            student.setDegrees(ns)
            for pInd, p in enumerate(num_samples_list):
                for lamInd, lam in enumerate(lams):
                    start_time = datetime.now(tz)
                    for trial in range(num_trials):
                        train_loader, test_loader = DatasetMaker.split_data(p, x, y)
                        if shuffle_degrees:
                            ns = EnsembleLib.genNs(k, M, connDist)
                            student.setDegrees(ns)
                        student.regress_readout(train_loader, test_loader, lam)
                        #errors = student.eval(train_loader, test_loader, auxFuncs.mean, auxFuncs.SquareError)
                        test_err[connDistInd, kInd, lamInd, pInd, trial] = student.calcEg_linear(sigma_s, w_star)
                    print('P = ' + str(p) +  '; Eg = ' + str(np.mean(test_err[connDistInd, kInd, lamInd, pInd, :]))+  '; dist = ' + connDist +  '; K = ' + str(k) +'; lam = ' + str(lam) + '; time: ' + str(time_diff(start_time, datetime.now(tz))))
           
    #Store results and everything needed for theory plots in output dict.        
    outputDict = {}
    outputDict['num_samples_list'] = num_samples_list
    outputDict['w_star'] = w_star
    outputDict['lams'] = lams
    #outputDict['tr_err'] = tr_err
    outputDict['test_err'] = test_err
    outputDict['Ordering'] = ['conDistInd', 'kInd', 'lamInd','pInd', 'trial']#The order of the indices in the tr_err and test_err tensors
    outputDict['projType'] = 'np'
    outputDict['sigma_s'] = sigma_s
    outputDict['sigma_0'] = sigma_0
    outputDict['eta'] = eta
    outputDict['zeta'] = zeta
    outputDict['KVals'] = KVals
    outputDict['shuffle_degrees'] = shuffle_degrees
    np.save(exptPath+'/output_dict', outputDict)
    
#Simulate linear regression by a single readout with a variety of subsampling fractions.
#Average over randomly drawn weight vectors w_star
def runEvP_SingleReadout_AlterThresh(expt_name, sigma_s, sigma_0, nus, eta, zeta, lams, num_samples_list, num_trials, aveGTW = False):
    assert sigma_s.shape[0] == sigma_s.shape[1]
    assert sigma_s.shape == sigma_0.shape
        
    M = sigma_s.shape[0]
    numPVals = len(num_samples_list)
    numNuVals = len(nus)
    numLamVals = len(lams)
    
    exptPath = expt_name
    print(expt_name)
    if not os.path.exists(exptPath):
        # Create a new directory because it does not exist 
        os.makedirs(exptPath)
        print("The new experiment directory is created!")
    else:
        print("The experiment directory already exists!")
        
    test_err = np.empty([numNuVals, numLamVals, numPVals, num_trials])
    ns = EnsembleLib.genNs(1, M) #will be replaced before running anything.
    student = EnsembleLib.MultiReadout(M, ns, sigma_0, eta, auxFuncs.identity, align_proj = True)
    
    #Starts with randomly drawn ground truth weights.
    w_star =  torch.normal(torch.zeros(M), torch.ones(M)).to('cuda')
    D = 10000
    x, y = DatasetMaker.makeGaussianDataset_lin(D, w_star, sigma_s, zeta)
    
    for pInd, p in enumerate(num_samples_list):
        for nuInd, nu in enumerate(nus):
            start_time = datetime.now(tz)
            for trial in range(num_trials):
                ns = [int(M*nu)]
                student.setDegrees(ns)
                w_star =  torch.normal(torch.zeros(M), torch.ones(M)).to('cuda')    
                x, y = DatasetMaker.makeGaussianDataset_lin(int(p), w_star, sigma_s, zeta)
                train_loader, test_loader = DatasetMaker.split_data(p, x, y)
                for lamInd, lam in enumerate(lams):
                    student.regress_readout(train_loader, test_loader, lam)
                    #errors = student.eval(train_loader, test_loader, auxFuncs.mean, auxFuncs.SquareError)
                    test_err[nuInd, lamInd, pInd, trial] = student.calcEg_linear(sigma_s, w_star)
            print('P = ' + str(p) +  '; Eg = ' + str(np.mean(test_err[nuInd, :, pInd, :]))+  '; nu = ' + str(nu) +  '; P = ' + str(p) + '; time: ' + str(time_diff(start_time, datetime.now(tz))))

    #Store results and everything needed for theory plots in output dict.        
    outputDict = {}
    outputDict['num_samples_list'] = num_samples_list
    outputDict['lams'] = lams
    #outputDict['tr_err'] = tr_err
    outputDict['test_err'] = test_err
    outputDict['Ordering'] = ['nuInd', 'lamInd','pInd', 'trial']#The order of the indices in the tr_err and test_err tensors
    outputDict['projType'] = 'np'
    outputDict['sigma_s'] = sigma_s
    outputDict['sigma_0'] = sigma_0
    outputDict['eta'] = eta
    outputDict['zeta'] = zeta
    outputDict['nus'] = nus
    outputDict['aveGTW'] = aveGTW
    if aveGTW==False:
        outputDict['w_star'] = w_star
    np.save(exptPath+'/output_dict', outputDict)