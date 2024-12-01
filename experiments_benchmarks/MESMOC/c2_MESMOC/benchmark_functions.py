import math
import numpy as np
#from scipy.interpolate import interp1d
from copy import deepcopy


def Currin(x, d):
    return -1*float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))
def Currin_c(x, d):
    return -1*float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20)))) + 6

def branin(x1,d):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return -1*float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)

def branin_c(x1,d):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return -1*float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10) + 20

import numpy as np
import torch
from botorch.test_functions.multi_objective import C2DTLZ2
d = 4
M = 2
problem = C2DTLZ2(dim=d, num_objectives=M, negate=True).to(torch.float64)

def c2(X,d):
    X = torch.tensor([X])
    obj = problem(X.to(torch.float64)).squeeze(0).tolist()
    const =  problem.evaluate_slack(X.to(torch.float64)).squeeze(0).tolist()
    return obj + const
