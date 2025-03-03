# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
#import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,Matern
#import logging
from sklearn.gaussian_process.kernels import Kernel, RBF
import numpy as np

def batch_tanimoto_sim(x1: np.ndarray, x2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Tanimoto similarity between two arrays.
    Args:
        x1: `[n x d]` NumPy array.
        x2: `[m x d]` NumPy array.
        eps: Float for numerical stability.

    Returns:
        NumPy array denoting the Tanimoto similarity.
    """
    dot_prod = np.dot(x1, x2.T)
    x1_norm = np.sum(x1 ** 2, axis=-1, keepdims=True)
    x2_norm = np.sum(x2 ** 2, axis=-1, keepdims=True)

    tan_similarity = (dot_prod + eps) / (eps + x1_norm + x2_norm.T - dot_prod)
    
    return np.maximum(tan_similarity, 0)

class TanimotoKernel(Kernel):
    """
    Custom Tanimoto Kernel for GaussianProcessRegressor in scikit-learn.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.length_scale =1.0
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Calculate the kernel matrix between X and Y using the Tanimoto similarity.
        """
        if Y is None:
            Y = X
        K = batch_tanimoto_sim(X, Y, eps=self.eps)

        # No gradients for now
        if eval_gradient:
            return K, np.zeros_like(K)
        return K

    def diag(self, X):
        """Return the diagonal of the kernel matrix."""
        return np.ones(X.shape[0])

    def is_stationary(self):
        """Tanimoto kernel is not stationary."""
        return False

class GaussianProcess:
    def __init__(self, dim):
        self.dim = dim
        self.kernel =  TanimotoKernel()
        self.beta=1e6
        self.xValues = []
        self.yValues = []
        self.yValuesNorm=[]
        self.model = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=5)
#        self.model = GaussianProcessRegressor(kernel=self.kernel,normalize_y=True,n_restarts_optimizer=5)

        
    def fitNormal(self):
        y_mean = np.mean(self.yValues)
        y_std = self.getstd()
        self.yValuesNorm= (self.yValues - y_mean)/y_std
        self.model.fit(self.xValues, self.yValuesNorm)
    def fitModel(self):
        self.model.fit(self.xValues, self.yValues)

    
    def addSample(self, x, y):
        self.xValues.append(x)
        self.yValues.append(y)

    def getPrediction(self, x):
        mean, std = self.model.predict(x.reshape(1,-1),return_std=True)
        if std[0]==0:
            std[0]=np.sqrt(1e-5)*self.getstd()
        return mean, std
    def getmeanPrediction(self, x):
        mean = self.model.predict(x.reshape(1,-1))
        return mean
    def getmean(self):
        return np.mean(self.yValues)
    def getstd(self):
        y_std=np.std(self.yValues)
        if y_std==0:
            y_std=1
        return y_std
    