##---A penicillin function suitable for `platypus` package---##
#modified from https://github.com/HarryQL/TuRBO-Penicillin/blob/main/BO-Penicillin/Penicilin_Demo_complex-BO.ipynb
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
import numpy as np
def peni(X_input):
    
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
    return [P, -CO2, -t], [P, -CO2, -t]
