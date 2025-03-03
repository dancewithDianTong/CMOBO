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

def design(x, d):
    lb = np.array([55.0, 75.0, 1000.0, 11.0])
    ub = np.array([80.0, 110.0, 3000.0, 20.0])
    #denormalize 
    x = np.array(x)
    x = lb + (ub - lb) * x

    # Extract variables
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    # First objective function (n x 1)
    f1 = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1.0)
    
    # Second objective function (n x 1)
    f2 = (9.82e6 * (x2**2 - x1**2)) / (x3 * x4 * (x2**3 - x1**3))

    # Original constraints (n x 4)
    g1 = (x2 - x1) - 20.0
    g2 = 0.4 - (x3 / (3.14 * (x2**2 - x1**2)))
    g3 = 1.0 - (2.22e-3 * x3 * (x2**3 - x1**3)) / ((x2**2 - x1**2) ** 2)
    g4 = (2.66e-2 * x3 * x4 * (x2**3 - x1**3)) / (x2**2 - x1**2) - 900.0

    # Stack the constraints (n x 4)
    # g = [g1, g2, g3, g4]

    # Return negative of the objectives and constraints
    return [-f1, -f2, g1, g2, g3, g4]