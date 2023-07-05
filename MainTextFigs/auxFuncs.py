import torch
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
tz = pytz.timezone('US/Eastern')

#Single Site readout (non)linearities:
def identity(x):
    return x

def sign(x):
    return torch.sign(x)

#Ensemble readout functions:
def mean(x):
    return torch.mean(x, dim = 0)

def majorityVote(x):
    return torch.sign(torch.mean(torch.sign(x), dim = 0))

def scoreAverage(x):
    return torch.sign(torch.mean(x, dim = 0))

#Error Functions:
def SquareError(out,y):
    return torch.mean(torch.square(out-y))

def SgnErrorRate(out,y):
    return 1/4*torch.mean(torch.square(out-y))


#Functions for Timing Operations
def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

#Matrix square root.
def matrix_sqrt(A):
    L,V = torch.linalg.eigh(A)
    DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.T

def makeCorrelatedMatrix(M, c):
    return (1-c)*torch.eye(M)+c*torch.ones(M)