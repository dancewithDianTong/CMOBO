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


#peni
def peni(X_input,d):
    #assume bound to be [0,1]
    #apply denormalize first
    lower_bounds = np.array([60, 10, 293, 10, 0.01, 600, 5])
    upper_bounds = np.array([120, 18, 303, 18, 0.1, 700, 6.5])
    X_input = np.array(X_input)
    X_input = lower_bounds + (upper_bounds - lower_bounds) * X_input
    C_L_star = 8.26

    Y_xs = 0.45
    Y_xo = 0.04
    Y_ps = 0.90
    Y_po = 0.20



    K_1 = 10**(-10)
    K_2 = 7 * 10**(-5)
    m_X = 0.014
    m_o = 0.467

    alpha_1 = 0.143
    alpha_2 = 4*10**(-7)
    alpha_3 = 10**(-4)
    mu_X = 0.092
    K_X = 0.15
    # K_ox = 2*10**(-2)
    # K_op = 5*10**(-4)
    mu_p = 0.005
    K_p = 0.0002
    K_I = 0.10
    p = 3
    K = 0.04
    k_g = 7 * 10**(3)
    E_g = 5100
    k_d = 10**(33)
    E_d = 50000

    # rou_dot_C_p = 1/1500
    # rou_c_dot_C_pc = 1/2000

    rou_dot_C_p = 1000
    rou_c_dot_C_pc = 1000


    r_q1 = 60
    r_q2 = 1.6783 * 10**(-4)
    a = 1000
    b = 0.60

    alpha = 70
    beta = 0.4
    lambd = 2.5 * 10**(-4)
    gamma = 10**(-5)


    # kelvin
    T_v = 273
    T_o = 373



    # CAL/(MOL K)
    R = 1.9872
    
#     V_limits, X_limits, T_limits, S_limits, F_limits, s_f_limits, H_limits
    
    V, X, T, S, F, s_f, H_ = X_input[0],X_input[1],X_input[2],X_input[3], X_input[4], X_input[5], X_input[6]
    
    P = 0
    CO2 = 0
    t = 0
    dt = 1
    H = 10**(-H_)

    for i in range(2500):
        
        F_loss = V * lambd*(np.exp(5*((T - T_o)/(T_v - T_o))) - 1)
        dV_dt = F  - F_loss

        mu = (mu_X / (1 + K_1/H + H/K_2)) * (S / (K_X * X + S))  * ((k_g * np.exp(-E_g/(R*T))) - (k_d * np.exp(-E_d/(R*T))))
        dX_dt = mu * X - (X / V) * dV_dt
        
        mu_pp = mu_p * (S / (K_p + S + S**2 / K_I)) 
        dS_dt = - (mu / Y_xs) * X - (mu_pp/ Y_ps) * X - m_X * X + F * s_f / V - (S / V) * dV_dt
        
        dP_dt = (mu_pp * X) - K * P - (P / V) * dV_dt    
        
        dCO2_dt = alpha_1 *dX_dt + alpha_2 * X + alpha_3


        # UPDATE
        t += dt
        P = P + dP_dt*dt
        V = V + dV_dt*dt
        X = X + dX_dt*dt
        S = S + dS_dt*dt
        CO2 = CO2 + dCO2_dt*dt
        

        if V > 180:
#             print('Too large V')
            break

        if S < 0:
#             print('Too small S')
            break

        if dP_dt < 10e-12:
#             print('Converged P')
            break

#     print('final results: ' + 'P = '+str(np.round(P, 2)) +', S = '+str(np.round(S, 2)) + ', X = ' + str(np.round(X, 2)) + ', V = ' + str(np.round(V, 2)) + ', t = ' + str(i))
#     GpyOpt does minimization only
    return [P, -CO2, -t, P-10, -CO2+60, -t+350]

