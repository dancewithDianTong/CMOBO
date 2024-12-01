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

def f_1(x,d):
    r = 1/x[0]+ x[1]
    return float(-r)
def f_1_c(x,d):
    r = 1/x[0]+ x[1]
    return float(-r) +1.9

def f_2(x,d):
    r = x[0]+ x[1]**2
    return float(-r) 
def f_2_c(x,d):
    r = x[0]+ x[1]**2
    return float(-r) + 2.25
def toy(x,d):
    r1 = -(1/(x[0])+ x[1])
    r2 = -(x[0]+ x[1]**2)
    return [r1, r2] + [r1 + 1.9, r2+ 2.25]