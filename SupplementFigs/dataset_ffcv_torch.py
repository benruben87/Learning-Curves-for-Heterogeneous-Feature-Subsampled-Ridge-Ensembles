import os

import numpy as np
import torch
import torchvision

def loader_to_tensor(loader, numpy=False):
    X = []
    Y = []
    for x, y in loader:
        X.append(x.cpu().numpy())
        Y.append(y.cpu().numpy())
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y


def create_sorted_dataset(N_tr, dataset_name, flatten=False, to_onehot=False, dtype=torch.float, sort=False, device='cuda'):
    from torchvision.datasets import MNIST, CIFAR10
    if dataset_name == 'mnist':
        input_dim = 28*28
        num_classes = 10
        train_set = MNIST('./datasets', train=True, download=True)
        test_set = MNIST('./datasets', train=False, download=True)
        x_train, y_train = (np.array(train_set.data)[:N_tr], np.array(train_set.targets).squeeze()[:N_tr])
        x_test, y_test = (np.array(test_set.data), np.array(test_set.targets).squeeze())
    if dataset_name == 'cifar10':
        input_dim = 32*32*3
        num_classes = 10
        train_set = CIFAR10('./datasets', train=True, download=True)
        test_set = CIFAR10('./datasets', train=False, download=True)
        x_train, y_train = (np.array(train_set.data)[:N_tr], np.array(train_set.targets).squeeze()[:N_tr])
        x_test, y_test = (np.array(test_set.data), np.array(test_set.targets).squeeze())
    
    if sort:
        idx = np.argsort(y_train)
        x_train = x_train[idx]
        y_train = y_train[idx]

        idx = np.argsort(y_test)
        x_test = x_test[idx]
        y_test = y_test[idx]
        
#     if flatten:
        
    if to_onehot:
        y_train = np.eye(num_classes)[y_train]
        y_test = np.eye(num_classes)[y_test]
        
        
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
        
    x_train = x_train/255.0
#     x_train -= x_train.mean(axis=1, keepdims = True)
#     x_train = x_train / np.linalg.norm(x_train, axis=1, keepdims = True)
    
    x_test = x_test/255.0
#     x_test -= x_test.mean(axis=1, keepdims = True)
#     x_test = x_test / np.linalg.norm(x_test, axis=1, keepdims = True)
    
    return (torch.tensor(x_train, device=device, dtype=dtype), 
            torch.tensor(y_train, device=device, dtype=dtype), 
            torch.tensor(x_test, device=device, dtype=dtype), 
            torch.tensor(y_test, device=device, dtype=dtype), 
            input_dim, num_classes)
