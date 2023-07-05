#Feature Subsampling Library.
#Make and train deep neural networks
#Re-train readouts and ensembles of readouts
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
from torch.cuda.amp import GradScaler, autocast
import torchvision
import copy

from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

from scipy.stats import gamma

tz = pytz.timezone('US/Eastern')

def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

class Dense(nn.Linear):
    def __init__(self, in_features, out_features, act='linear', shift = 0, sigma_w=1., sigma_b=0, ntk=True, device = 'cuda'):
        
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.ntk = ntk
        self.act = act
        self.shift = shift
        
        if sigma_b == 0:
            bias = False
        
        super(Dense, self).__init__(in_features, out_features, bias)
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.sigma_b != 0:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        weight = self.sigma_w*self.weight
        if self.ntk:
            weight = weight/np.sqrt(self.in_features)
            
        if self.sigma_b != 0:
            bias = self.sigma_b * self.bias
        else:
            bias = None
            
        out = F.linear(input, weight, bias)
        if self.act=='relu':
            out = F.relu(out+self.shift)
            
        return out

def makeFeatureMap(input_dim, hidden_width_list, act='relu', ntk=True, shift = 0):

    hidden_width_list = np.append(input_dim, hidden_width_list)
    depth = len(hidden_width_list)

    layers = []
    for i in range(depth-1):
        layers += [(f'feature_{i+1}', Dense(hidden_width_list[i], hidden_width_list[i+1], act=act, ntk=ntk))]

    featureMap = torch.nn.Sequential(OrderedDict(layers)).cuda()
    return featureMap

def makeFeatureMaps_ShiftList(input_dim, hidden_width_list, act='relu', ntk=True, shifts = [0]):

    hidden_width_list = np.append(input_dim, hidden_width_list)
    depth = len(hidden_width_list)
    featureMaps = []
    
    for i in range(depth-1):
        layers += [(f'feature_{i+1}', Dense(hidden_width_list[i], hidden_width_list[i+1], act=act, ntk=ntk))]
        featureMap_NoShift = torch.nn.Sequential(OrderedDict(layers)).cuda()
    
    for shift in shifts:
        featureMap = copy.deepcopy(featuremap_NoShift)
        featureMap.shift = shift
        featureMaps.append(featureMap)
    return featureMaps
    
class FeatureNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_width_list, act='relu', ntk=True, device = 'cuda'):
        super(FeatureNet,self).__init__()
        
        self.multiReadout = False
        self.device = device
        self.feature_map = makeFeatureMap(input_dim, hidden_width_list, act = act, ntk = ntk, shift = 0)
        self.readout = Dense(hidden_width_list[-1], num_classes, ntk = ntk).to(device)
        
        #Make sure to train the features
#         for param in self.feature_map.parameters():
#             param.requires_grad = False

    def forward(self, x):
        features = self.feature_map(x)
        output= self.readout(features)
        return output
    
class Proj(torch.nn.Module):
        def __init__(self, proj_mat):
            super(Proj, self).__init__()
            self.proj = proj_mat
        def forward(self, x): 
            if len(self.proj.shape)==1:
                return x * self.proj #For backwards compatibility: used if vector mask is passed in
            elif len(self.proj.shape)==2:
                return x@torch.t(self.proj)
            else:
                raise Exception("Projection matrix must have 1 or 2 dimensions.")
    
def makeProjectionMatrix(proj_ratio, num_features, align_proj, smart_proj = False, features = None):
    
    #ToDo: Add code that produces a "smart" projection matrix which selects the features that are most often activated.
    #Later: Try instead learning a smart projection using monte carlo sampling of the projection vector, which consists of zeros and 1's in a fixed ratio.
    
    k = int(proj_ratio*num_features)
    #If align_proj is true, projects to a random selection of neurons
    if align_proj:
        mask = torch.randperm(num_features, device='cuda', requires_grad=False)[:k]
        proj_mat = torch.eye(num_features, device = 'cuda')[mask]
    #If align_proj is false, generates a random gaussian projection matrix.
    else:
        proj_mat = 1/torch.sqrt(torch.tensor(k, device='cuda'))*torch.zeros(size=[k, num_features], device='cuda').normal_(mean=0, std=1)
    return proj_mat

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
def genGaussianAlpha(K, stdFra=1/3):
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

class StudentNetwork(nn.Module):
    def __init__(self, feature_map, num_classes, proj_ratios, ntk = True, align_proj = True, shift = None, device = 'cuda', OutStrategy = 'mean'):
        
        super().__init__()
        self.device = device
        self.multiReadout = True
        #Alters feature map to include shift if shift is passed in as argument
        if shift !=None:
            feature_map[-1].shift = shift
            
        self.numReadoutReps = proj_ratios.shape[0]
        num_features = feature_map[-1].weight.shape[0]
        
        proj_list = []
        readout_list = []
        
        for i in range(self.numReadoutReps):
            proj_list.append(Proj(makeProjectionMatrix(proj_ratios[i], num_features, align_proj)).to(device))
            k = int(proj_ratios[i]*num_features)
            readout_list.append(Dense(k, num_classes, ntk=ntk).to(device))
        
        #Don't train the projection either.
        for proj in proj_list:
            for param in proj.parameters():
                param.requires_grad = False
        
        self.feature_map = feature_map
        self.proj_list = nn.ModuleList(proj_list)
        self.readout_list = nn.ModuleList(readout_list)
        self.OutStrategy = OutStrategy
        
        #Don't train the feature map.
        for param in feature_map.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        features = self.feature_map(x)
        features_proj = [self.proj_list[i](features) for i in range(self.numReadoutReps)]
        outputs = torch.stack([self.readout_list[i](self.proj_list[i](features)) for i in range(self.numReadoutReps)])
        if self.OutStrategy == 'mean':
            collective_outputs = torch.mean(outputs, dim = 0)
        elif self.OutStrategy == 'median':
            collective_outputs = torch.median(outputs, dim = 0)[0]
        elif self.OutStrategy == 'threshmean':
            threshOutputs = (1+torch.tanh(5*(outputs-.5)))/2
            collective_outputs = torch.mean(threshOutputs, dim = 0)
        else:
            raise Exception('readout strategy must be mean, median, or threshmean')
        return outputs, features_proj, collective_outputs

class CustomModel:

    def __init__(self, model, binary=False):
        self.model = model
        self.binary = binary
        self.callback = None

    def compile(self, opt, loss_fn, callback, scheduler=None):
        self.opt = opt
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.callback = callback

    def train_step(self, x, y):
        self.opt.zero_grad(set_to_none=True)
        with autocast():
            # Predict and compute loss
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            logs = {'loss': loss.detach().cpu().numpy()}
            
        ## Backprop
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scheduler is not None: self.scheduler.step()
            
        logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)

        return logs

    def fit(self, tr_loader, test_loader, epochs):

        self.callback.on_train_begin()
        for ep in range(epochs):
            ## Batch training loop
            for x, y in tr_loader:
                logs = self.train_step(x, y)
                self.callback.on_batch_end(logs)

            # Compute Model Accuracy
            model_acc = self.eval(tr_loader, test_loader, training=True)
            model_loss = self.calcLoss(tr_loader, test_loader, training=True)

            self.callback.on_epoch_end(model_acc, model_loss)
        self.callback.on_train_end()       
        
    #Method to fit only the readout weights using linear regression.
    #Assumes training loader and test loader store a single (full) batch
    def regress_readout(self, tr_loader, test_loader):
        self.callback.on_train_begin()
        prefactor = self.model[-1].sigma_w
        if self.model[-1].ntk:
            prefactor = prefactor/torch.sqrt(torch.tensor(self.model[-1].weight.shape[1]))
        
        psiT = self.model.feature_map(tr_loader[0][0])#transpose of NFeat by P matrix of features
        psi = psiT.t()#NFeat by P matrix of features
        YT = tr_loader[0][1] # Nout by P matrix of targets
        #if features.shape[0]>features.shape[1]:
        #    W = targets@torch.linalg.pinv(features_T@features)@features_T
        #else:
        #    W = targets@features_T@torch.linalg.pinv(features@features_T)
        W =torch.t(torch.linalg.pinv(prefactor*psiT)@(YT))
        with torch.no_grad():
            self.model[-1].weight = nn.Parameter(W)
        
        with autocast():
            # Predict and compute loss
            y_pred = self.model(tr_loader[0][0])
            loss = self.loss_fn(y_pred, tr_loader[0][1])
            logs = {'loss': loss.detach().cpu().numpy()}
        
        logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)
        self.callback.on_batch_end(logs)
        
        model_acc = self.eval(tr_loader, test_loader, training=True)
        model_loss = self.calcLoss(tr_loader, test_loader, training=True)
        self.callback.on_epoch_end(model_acc, model_loss)
        self.callback.on_train_end()
    
    #Method to fit only the readout weights according to the hebbian perscription.
    #Assumes training loader and test loader store a single (full) batch
    def badhebb_readout(self, tr_loader, test_loader):
        self.callback.on_train_begin()
        P = tr_loader[0][0].shape[0]
        prefactor = (self.model[-1].sigma_w*P)**(-1)
        if self.model[-1].ntk:
            prefactor = prefactor*torch.sqrt(torch.tensor(self.model[-1].weight.shape[1]))
        
        psiT = self.model.feature_map(tr_loader[0][0])#transpose of NFeat by P matrix of features
        Y = tr_loader[0][1].t() # Nout by P matrix of targets
        W = prefactor*Y@psiT
        with torch.no_grad():
            self.model[-1].weight = nn.Parameter(W)
        
        with autocast():
            # Predict and compute loss
            y_pred = self.model(tr_loader[0][0])
            loss = self.loss_fn(y_pred, tr_loader[0][1])
            logs = {'loss': loss.detach().cpu().numpy()}
        
        logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)
        self.callback.on_batch_end(logs)
        
        model_acc = self.eval(tr_loader, test_loader, training=True)
        model_loss = self.calcLoss(tr_loader, test_loader, training=True)
        self.callback.on_epoch_end(model_acc, model_loss)
        self.callback.on_train_end()

    #Method to fit only the readout weights according to the hebbian perscription.
    #Assumes training loader and test loader store a single (full) batch
    def hebbCenter_readout(self, tr_loader, test_loader):
        self.callback.on_train_begin()
        P = tr_loader[0][0].shape[0]
        prefactor = (self.model[-1].sigma_w*P)**(-1)
        if self.model[-1].ntk:
            prefactor = prefactor*torch.sqrt(torch.tensor(self.model[-1].weight.shape[1]))
        
        psiT = self.model.feature_map(tr_loader[0][0])#transpose of NFeat by P matrix of features
        Y = tr_loader[0][1].t() # Nout by P matrix of targets
        psiT_mean = torch.outer(torch.ones(psiT.shape[0], device = 'cuda'), torch.mean(psiT, dim = 0))
        W = prefactor*Y@(psiT - psiT_mean)
        with torch.no_grad():
            self.model[-1].weight = nn.Parameter(W)
        
        with autocast():
            # Predict and compute loss
            y_pred = self.model(tr_loader[0][0])
            loss = self.loss_fn(y_pred, tr_loader[0][1])
            logs = {'loss': loss.detach().cpu().numpy()}
        
        logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)
        self.callback.on_batch_end(logs)
        
        model_acc = self.eval(tr_loader, test_loader, training=True)
        model_loss = self.calcLoss(tr_loader, test_loader, training=True)
        self.callback.on_epoch_end(model_acc, model_loss)
        self.callback.on_train_end()

    def eval(self, tr_loader, test_loader, training=False):
        self.model.eval()
        with torch.no_grad():
            pred_acc_train, total_num = 0., 0.
            for x, y in tr_loader:
                with autocast():
                    out =self.model(x)
                    
                    if self.binary:
                        out = 2*(out > 0)-1
                        pred_acc_train += out.eq(y).sum().cpu().item()
                    else:
                        pred_acc_train += out.argmax(1).eq(y.argmax(1)).sum().cpu().item()
                        
                    total_num += x.shape[0]
            pred_acc_train = pred_acc_train / total_num * 100

            pred_acc_test, total_num = 0., 0.
            for x, y in test_loader:
                with autocast():
                    out =self.model(x)
                    
                    if self.binary:
                        out = 2*(out > 0)-1
                        pred_acc_test += out.eq(y).sum().cpu().item()
                    else:
                        pred_acc_test += out.argmax(1).eq(y.argmax(1)).sum().cpu().item()
                        
                    total_num += x.shape[0]
            pred_acc_test = pred_acc_test / total_num * 100
        ## Compute accuracies (self.eval turns off batch_norm etc. So turn them on after evaluating by .train() method) 
        if training:
            self.model.train()

        return (pred_acc_train, pred_acc_test)
    
    def calcLoss(self, tr_loader, test_loader, training=False):
        self.model.eval() 
        with torch.no_grad():
            with autocast():
                # Predict and compute loss
                y_pred = self.model(tr_loader[0][0])
                loss = self.loss_fn(y_pred, tr_loader[0][1]).detach().cpu().numpy()
                y_pred = self.model(test_loader[0][0])
                test_loss = self.loss_fn(y_pred, test_loader[0][1]).detach().cpu().numpy()
        if training:
            self.model.train()
        return (loss, test_loss)

class CustomCallback:

    def __init__(self, num_classes, verbose = True):
        
        self.num_classes = num_classes
        self.epoch_count = 0
        self.batch_count = 0
        self.losses = []
        self.test_losses=[]
        self.readout_weights = []
        self.tr_acc = []
        self.test_acc = []
        self.verbose = verbose

    def on_train_begin(self):
        self.t_start = datetime.now(tz)
        if self.verbose:
            print('Training Started... %s' % self.t_start.ctime(), flush=True)

    def on_batch_end(self, logs):
        
        ## Reinitialize storing arrays
        if self.batch_count == 0:
            self.readout_weights_batch = []

        self.losses.append(logs['loss'])
        self.readout_weights_batch += [logs['readout_weights']]

        self.batch_count += 1

    def on_epoch_end(self, model_acc, model_loss = None):
        self.batch_count = 0
        self.epoch_count += 1
        ## Compute accuracies
        tr_acc, test_acc = model_acc
        self.tr_acc += [tr_acc]
        self.test_acc += [test_acc]
        
        if model_loss != None:
            tr_loss, test_loss = model_loss
            self.losses+=[tr_loss]
            self.test_losses += [test_loss]

        self.readout_weights += [np.array(self.readout_weights_batch).mean(0)]

        if self.verbose:
            prt_str = 'Epoch: %d | Loss -- Tr: %.3f | ' % (self.epoch_count, self.losses[-1])
            if model_loss!=None:
                prt_str += 'Tr/Test Acc: %.1f, %.1f | Tr/Test Loss: %.3f, %.3f | dt: %s ----' % (tr_acc, test_acc, tr_loss, test_loss, time_diff(self.t_start, datetime.now(tz)))
            else:
                prt_str += 'Tr/Test Acc: %.1f, %.1f | dt: %s ----' % (tr_acc, test_acc, time_diff(self.t_start, datetime.now(tz)))

            print(prt_str, end="\r", flush=True)

    def on_train_end(self):
        self.losses = np.array(self.losses)
        self.test_losses = np.array(self.test_losses)
        self.tr_acc = np.array(self.tr_acc)
        self.test_acc = np.array(self.test_acc)
        self.readout_weights = np.array(self.readout_weights)

        if self.verbose:
            print('\n Training finished! Total time: %s' % time_diff(self.t_start, datetime.now(tz)), flush=True)

    def get_history(self):
        return (self.losses, self.tr_acc, self.test_acc, self.readout_weights)
    
class CustomMultiReadoutModel:
    def __init__(self, model, binary=False):
        self.model = model
        self.binary = binary
        self.callback = None

    def compile(self, opt, loss_fn, callback, scheduler=None):
        self.opt = opt
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.callback = callback

    def train_step(self, x, y):
        self.opt.zero_grad(set_to_none=True)
        with autocast():
            # Predict and compute loss
            y_preds, features, mean_ypred = self.model(x)
            loss = 0
            for i in range(self.model.numReadoutReps):
                loss+= self.loss_fn(y_preds[i], y)
            logs = {'loss': loss.detach().cpu().numpy()}
        
        ## Backprop
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scheduler is not None: self.scheduler.step()
            
        #logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)

        return logs

    def fit(self, tr_loader, test_loader, epochs):

        self.callback.on_train_begin()
        for ep in range(epochs):
            ## Batch training loop
            for x, y in tr_loader:
                logs = self.train_step(x, y)
                self.callback.on_batch_end(logs)

            # Compute Model Accuracy
            model_acc, model_err = self.eval(tr_loader, test_loader, training=True)
            model_loss = self.calcLoss(tr_loader, test_loader, training=True)

            self.callback.on_epoch_end(model_acc, model_err, model_loss)
        self.callback.on_train_end()
        
    #Method to fit only the readout weights using linear regression.
    #Assumes training loader and test loader store a single (full) batch
    def regress_readout(self, tr_loader, test_loader):
        self.callback.on_train_begin()
        
        for i in range(self.model.numReadoutReps):
            prefactor = self.model.readout_list[i].sigma_w
            if self.model.readout_list[i].ntk:
                prefactor = prefactor/torch.sqrt(torch.tensor(self.model.readout_list[i].weight.shape[1]))

            psiT = self.model.proj_list[i](self.model.feature_map(tr_loader[0][0]))#transpose of NFeat by P matrix of features
            psi = psiT.t()#NFeat by P matrix of features
            YT = tr_loader[0][1] # Nout by P matrix of targets
            #if features.shape[0]>features.shape[1]:
            #    W = targets@torch.linalg.pinv(features_T@features)@features_T
            #else:
            #    W = targets@features_T@torch.linalg.pinv(features@features_T)
            W =torch.t(torch.linalg.pinv(prefactor*psiT)@(YT))
            with torch.no_grad():
                self.model.readout_list[i].weight = nn.Parameter(W)

        with autocast():
            # Predict and compute loss
            y_preds, features, y_pred_mean = self.model(tr_loader[0][0])
            loss = self.loss_fn(y_pred_mean, tr_loader[0][1])
            logs = {'loss': loss.detach().cpu().numpy()}

        #logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)
        self.callback.on_batch_end(logs)
        
        model_acc, model_err = self.eval(tr_loader, test_loader, training=True)
        model_loss = self.calcLoss(tr_loader, test_loader, training=True)
        self.callback.on_epoch_end(model_acc, model_err, model_loss)
        self.callback.on_train_end()
        
    #Method to fit only the readout weights according to the hebbian perscription.
    #Assumes training loader and test loader store a single (full) batch
    def hebbCenter_readout(self, tr_loader, test_loader):
        self.callback.on_train_begin()
        
        for i in range(self.model.numReadoutReps):
            prefactor = self.model.readout_list[i].sigma_w
            if self.model.readout_list[i].ntk:
                prefactor = prefactor/torch.sqrt(torch.tensor(self.model.readout_list[i].weight.shape[1]))

            psiT = self.model.proj_list[i](self.model.feature_map(tr_loader[0][0]))#transpose of NFeat by P matrix of features
            psi = psiT.t()#NFeat by P matrix of features
            Y = tr_loader[0][1].T # Nout by P matrix of targets
            psiT_mean = torch.outer(torch.ones(psiT.shape[0], device = 'cuda'), torch.mean(psiT, dim = 0))
            W = prefactor*Y@(psiT - psiT_mean)
            with torch.no_grad():
                self.model.readout_list[i].weight = nn.Parameter(W)

        with autocast():
            # Predict and compute loss
            y_preds, features, y_pred_mean = self.model(tr_loader[0][0])
            loss = self.loss_fn(y_pred_mean, tr_loader[0][1])
            logs = {'loss': loss.detach().cpu().numpy()}

        #logs['readout_weights'] = self.model.readout.weight.detach().cpu().numpy().reshape(-1)
        self.callback.on_batch_end(logs)
        
        model_acc, model_err = self.eval(tr_loader, test_loader, training=True)
        model_loss = self.calcLoss(tr_loader, test_loader, training=True)
        self.callback.on_epoch_end(model_acc, model_err, model_loss)
        self.callback.on_train_end()

    def eval(self, tr_loader, test_loader, training=False):
        self.model.eval()
        with torch.no_grad():
            pred_acc_train, total_num = 0., 0.
            for x, y in tr_loader:
                with autocast():
                    out_reps, out_features_reps, out = self.model(x)
                    
                    if self.binary:
                        out = 2*(out > 0)-1
                        pred_acc_train += out.eq(y).sum().cpu().item()
                    else:
                        pred_acc_train += out.argmax(1).eq(y.argmax(1)).sum().cpu().item()
                        
                    err_train = self.loss_fn(out, y)    
                    total_num += x.shape[0]
            pred_acc_train = pred_acc_train / total_num * 100

            pred_acc_test, total_num = 0., 0.
            for x, y in test_loader:
                with autocast():
                    out_reps, out_features_reps, out =self.model(x)
                    
                    if self.binary:
                        out = 2*(out > 0)-1
                        pred_acc_test += out.eq(y).sum().cpu().item()
                    else:
                        pred_acc_test += out.argmax(1).eq(y.argmax(1)).sum().cpu().item()
                    
                    err_test = self.loss_fn(out, y)
                    
                    total_num += x.shape[0]
            pred_acc_test = pred_acc_test / total_num * 100
        ## Compute accuracies (self.eval turns off batch_norm etc. So turn them on after evaluating by .train() method) 
        if training:
            self.model.train()

        return (pred_acc_train, pred_acc_test), (err_train, err_test)
    
    def calcLoss(self, tr_loader, test_loader, training=False):
        self.model.eval()
        with torch.no_grad():
            with autocast():
                # Predict and compute loss
                y_preds, features, meanYPred = self.model(tr_loader[0][0])
                loss = 0
                for i in range(self.model.numReadoutReps):
                    loss += self.loss_fn(y_preds[i], tr_loader[0][1]).detach().cpu().numpy()
                y_preds, features, meanYPred = self.model(test_loader[0][0])
                test_loss = 0
                for i in range(self.model.numReadoutReps):
                    test_loss += self.loss_fn(y_preds[i], test_loader[0][1]).detach().cpu().numpy()
        if training:
            self.model.train()
        return (loss, test_loss)

class CustomMultiReadoutCallback:

    def __init__(self, num_classes, verbose = True):
        
        self.num_classes = num_classes
        self.epoch_count = 0
        self.batch_count = 0
        self.losses = []
        self.test_losses=[]
        self.tr_err = []
        self.test_err = []
        self.tr_acc = []
        self.test_acc = []
        self.verbose = verbose

    def on_train_begin(self):
        self.t_start = datetime.now(tz)
        if self.verbose:
            print('Training Started... %s' % self.t_start.ctime(), flush=True)

    def on_batch_end(self, logs):
        
        ## Reinitialize storing arrays
        if self.batch_count == 0:
            self.readout_weights_batch = []

        self.losses.append(logs['loss'])
        self.batch_count += 1

    def on_epoch_end(self, model_acc, model_err = None, model_loss = None):
        self.batch_count = 0
        self.epoch_count += 1
        ## Compute accuracies
        tr_acc, test_acc = model_acc
        tr_err, test_err = model_err
        self.tr_acc += [tr_acc]
        self.test_acc += [test_acc]
        
        if model_loss != None:
            tr_loss, test_loss = model_loss
            self.losses+=[tr_loss]
            self.test_losses += [test_loss]
            
        if model_err !=None:
            tr_err, test_err = model_err
            self.tr_err+=[tr_err]
            self.test_err+= [test_err]

        if self.verbose:
            prt_str = 'Epoch: %d | Loss -- Tr: %.3f | ' % (self.epoch_count, self.losses[-1])
            if model_err!=None:
                prt_str += 'Tr/Test Err: %.1f, %.1f  |' % (tr_err, test_err)
            if model_loss!=None:
                prt_str += 'Tr/Test Acc: %.1f, %.1f  | Tr/Test Loss: %.3f, %.3f | dt: %s ----' % (tr_acc, test_acc, tr_loss, test_loss, time_diff(self.t_start, datetime.now(tz)))
            else:
                prt_str += 'Tr/Test Acc: %.1f, %.1f | dt: %s ----' % (tr_acc, test_acc, time_diff(self.t_start, datetime.now(tz)))

            print(prt_str, end="\r", flush=True)

    def on_train_end(self):
        self.losses = np.array(self.losses)
        self.test_losses = np.array(self.test_losses)
        self.tr_acc = np.array(self.tr_acc)
        self.test_acc = np.array(self.test_acc)
        #self.readout_weights = np.array(self.readout_weights)

        if self.verbose:
            print('\n Training finished! Total time: %s' % time_diff(self.t_start, datetime.now(tz)), flush=True)

    def get_history(self):
        return (self.losses, self.tr_acc, self.test_acc)