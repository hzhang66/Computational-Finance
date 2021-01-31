#MFE405 Computational Methods in Finance
#Project 6
#Author: Heyu Zhang



# python set up
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


#Question1
def StockPrices(S0, r, sd, T, paths, steps):
    
    dt = T/steps
    
    # Generate stochastic process and its antithetic paths
    Z = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    Z_inv = -Z
    
    dWt = math.sqrt(dt) * Z
    dWt_inv = math.sqrt(dt) * Z_inv
    
    # bind the normal and antithetic Wt
    dWt = np.concatenate((dWt, dWt_inv), axis=0)
    
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    
    for i in range (1, steps + 1):
        St[:, i] = St[:, i - 1]*np.exp((r - 1/2*np.power(sd, 2))*dt + sd*dWt[:, i - 1])
    
    return St[:, 1:]
def LookBackOption(S0, K, r, sd, T, paths, Type):
    
    # use daily frequency
    steps = 252
    St = StockPrices(S0, r, sd, T, paths, steps)
    
    # find the maximum and minimum stock price incurred for each path
    St_max = St.max(axis = 1)
    St_min = St.min(axis = 1)
    
    # find the lookback option value
    if Type == "call":
        option_value = np.exp(-r*T) * np.mean(np.maximum(St_max - K, 0))
    elif Type == "put":
        option_value = np.exp(-r*T) * np.mean(np.maximum(K - St_min, 0))
        
    return option_value

def q1(S0, X, T, r, sigma, N):
    np.random.seed(7)
    call_value = [LookBackOption(S0, X, r, i, T, N, "call") for i in sigma]
    put_value = [LookBackOption(S0, X, r, i, T, N, "put") for i in sigma]
    plt.figure(figsize=(10, 7))
    plt.plot(sigma, call_value,  linestyle='--', marker='o')

    plt.xlabel('Volitality $\sigma$')
    plt.ylabel('European Call Value $')
    plt.title('European Lookback Call Option Values')
    plt.savefig("Proj6_1a.jpg")
    plt.show()
    
    plt.figure(figsize=(10, 7))
    plt.plot(sigma, put_value,  linestyle='--', marker='o', color = 'orange')
    plt.xlabel('Volitality $\sigma$')
    plt.ylabel('European Put Value $')
    plt.title('European Lookback Put Option Values')
    plt.savefig("Proj6_1b.jpg")
    plt.show()
    
    
r = 0.03
N = 10000
X = 100
S0 = 98
T = 1
sigma = np.arange(0.12, 0.481, 0.04)

q1(S0, X, T, r, sigma, N)

#Question2
def loanCollateral_Vt(V0, mu, sigma, gamma, lambda1, T, paths):
    
    dt = 1/12
    steps = T * 12

    # Generate stochastic process and its antithetic paths
    Z = np.random.normal(0, 1, paths * steps).reshape((paths, steps))
    dWt = math.sqrt(dt) * Z

    # Initialize Vt process
    Vt = np.zeros((paths, steps + 1))
    Vt[:, 0] = V0

    # Build Vt Process
    for i in range (1, steps + 1):
        Vt[:, i] = (Vt[:, i - 1] * np.exp((mu - 1/2*np.power(sigma, 2)) * dt + sigma * dWt[:, i - 1]) 
                     * (1 + gamma * np.random.poisson(lambda1*dt, paths)))   
        
    return Vt[:, 1:]
def loanBalance_Lt(L0, r0, T, lambda2, delta, t):
    
    # find monthly APR (r)
    R = r0 + delta*lambda2
    r = R/12
    
    # find monthly payment
    n = T * 12
    PMT = (L0*r)/(1 - (1/(1+r)**n))
    
    # Find the loan value given time t
    a = PMT/r
    b = PMT/(r*(1 + r)**n)
    c = (1 + r)
    Lt = np.clip(a - b*c**(12*t), a_max = None, a_min = 0)
    
    return Lt[1:]
def recoveryRate_qt(alpha, eps, T, t):
    beta = (eps - alpha)/T
    qt = alpha + beta*t 
    return qt[1:]
def stopTime_Q(Vt, eps, paths, T):
    
    dt = 1/12
    t = np.arange(0, T + 0.01, 1/12)
    
    # find loan balance and recovery rate over time
    Lt = loanBalance_Lt(L0, r0, T, lambda2, delta, t)
    qt = recoveryRate_qt(alpha, eps, T, t)
    
    # find all of the time steps that the borrow are likely default, set the value of default_time at time step to 1
    residual_collateral = np.tile(Lt*qt, paths).reshape((paths, 12 * T))
    default_time = np.where(Vt - residual_collateral  <= 0, 1, 0)
    
    # Find stopping time Qt
    Q = np.argmax(default_time, axis = 1) 
    
    # Set all paths that has no default to the largest value of index
    no_default = np.where(np.sum(default_time, axis = 1) == 0)
    Q[no_default] = 5000
    
    return Q
def stopTime_S(lambda2, paths, T):
    
    # Generate a matrix of poisson process
    dt = 1/12
    Nt = np.clip(np.random.poisson(lambda2*dt, (paths, T*12)), 
                 a_max = 1,
                 a_min = 0)
    
    S = np.argmax(Nt, axis = 1) 
    # Set all paths that has no default to the largest value of index
    no_default = np.where(np.sum(Nt, axis = 1) == 0)
    S[no_default] = 5000
    
    return S
def estimated_default_time(Q, S):
    
    # find which one (Q, S) is exercised earlier
    qs = np.where(Q - S <= 0, 1, 0)
    
    # set the paths where there is no default to 5000
    no_default = np.where(Q + S == 5000*2)
    qs[no_default] = 5000
    
    return np.minimum(Q, S), qs
def loanModel(V0, mu, sigma, gamma, lambda1,
                       T, paths, L0, r0, lambda2, delta, eps):
    dt = 1/12
    t = np.arange(0, T + 0.01, 1/12)
    
    # find loan collateral value
    Vt = loanCollateral_Vt(V0, mu, sigma, gamma, lambda1, T, paths)
    
    # loan balance
    Lt = np.tile(loanBalance_Lt(L0, r0, T, lambda2, delta, t), paths).reshape((paths, 12 * T))
    
    # default time Q, S
    Q = stopTime_Q(Vt, eps, paths, T)
    S = stopTime_S(lambda2, paths, T)
    
    # optimal default time
    tau, qs = estimated_default_time(Q, S)
    
    # find which type of default occured
    default_q = np.where(qs == 1)
    default_s = np.where(qs == 0)
    no_default = np.where(qs == 5000)
    
    # find discount factor of each path
    df = np.exp(-r0*dt*tau) 

    # find payoff of the default option, based on conditions
    payoff = np.zeros(paths)
    payoff[default_q] = np.maximum(Lt[default_q, tau[default_q]] 
                                   - eps* Vt[default_q, tau[default_q]], 0)

    payoff[default_s] =  np.abs(Lt[default_s, tau[default_s]] 
                                - eps* Vt[default_s, tau[default_s]])

    payoff[no_default] = 0

    # discount the expected payoff and find the value of default option 
    option_value = np.mean(payoff*df)
    
    # find default intensity
    default_intensity =  1 - len(no_default[0])/paths
    
    # find expected stopping time
    expected_stoptime = np.mean(tau[np.where(tau != 5000)[0]]*dt)
    
    d = {'Option Value': round(option_value, 4),
        'Default Intensity': round(default_intensity, 4),
        'Expected Stop Time':round(expected_stoptime, 4)}
    
    return d

def Proj6_2func(lambda1, lambda2, T):
    V0 = 20000
    L0 = 22000
    mu = -0.1
    sigma = 0.2
    gamma = -0.4
    r0 = 0.02
    delta = 0.25
    alpha = 0.7
    eps = 0.95
    paths = 50000
    D=loanModel(V0, mu, sigma, gamma, lambda1,T, paths, L0, r0, lambda2, delta, eps)['Option Value']
    Prob=loanModel(V0, mu, sigma, gamma, lambda1,T, paths, L0, r0, lambda2, delta, eps)['Default Probability']
    Et=loanModel(V0, mu, sigma, gamma, lambda1,T, paths, L0, r0, lambda2, delta, eps)['Expected Exercise Time']
    return D, Prob, Et


def q2(seed):
    np.random.seed(seed)
    lambda1 = 0.2
    lambda2 = 0.4
    T = 5
    loan = pd.DataFrame(Proj6_2func(lambda1, lambda2, T), index = ['Option Value', 'Default Probability','Expected Exercise Time'])
    loan.columns = ['Values']
    l1 = np.arange(0.05, 0.45, 0.05)
    l2 = np.arange(0, 0.9, 0.1)
    Time = np.arange(3, 9, 1)
    lam2 = 0.4
    optionValue_l1 = []
    for lam1 in  l1:
            for T in Time:
                optionValue_l1.append(Proj6_2func(lam1, lam2, T)[0])
    optionValue_l1 = np.array(optionValue_l1).reshape((len(l1), len(Time)))

    lam1 = 0.2
    optionValue_l2 = []
    for lam2 in  l2:
            for T in Time:
                optionValue_l2.append(Proj6_2func(lam1, lam2, T)[0])
    optionValue_l2 = np.array(optionValue_l2).reshape((len(l2), len(Time)))

    plt.figure(figsize=(17, 7))
    plt.subplot(1,2,1)
    for i in range(len(optionValue_l1)):
        plt.plot(Time, optionValue_l1[i],  linestyle='--',
                 marker='o', label = '$\lambda1 = $' + str(l1[i]))

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Default Option Value ($)')
    plt.title('Default Option Value with Different $\lambda_1$')

    plt.subplot(1,2,2)

    for i in range(len(optionValue_l2)):
        plt.plot(Time, optionValue_l2[i],  linestyle='--',
                 marker='o', label = '$\lambda2 = $' + str(l2[i]))

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Default Option Value ($)')
    plt.title('Default Option Value with Different $\lambda_2$')
    plt.savefig("Proj6_2a.jpg")
    plt.show()



    lam2 = 0.4
    defaultIntensity_l1 = []
    for lam1 in  l1:
            for T in Time:
                defaultIntensity_l1.append(Proj6_2func(lam1, lam2, T)[1])
    defaultIntensity_l1 = np.array(defaultIntensity_l1).reshape((len(l1), len(Time)))
    lam1 = 0.2
    defaultIntensity_l2 = []
    for lam2 in  l2:
            for T in Time:
                defaultIntensity_l2.append(Proj6_2func(lam1, lam2, T)[1])
    defaultIntensity_l2 = np.array(defaultIntensity_l2).reshape((len(l2), len(Time)))

    plt.figure(figsize=(17, 7))
    plt.subplot(1,2,1)
    for i in range(len(defaultIntensity_l1)):
        plt.plot(Time, defaultIntensity_l1[i],  linestyle='--',
                 marker='o', label = '$\lambda1 = $' + str(l1[i]))

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Default Probability')
    plt.title('Default Probability with Different $\lambda_1$')

    plt.subplot(1,2,2)

    for i in range(len(defaultIntensity_l2)):
        plt.plot(Time, defaultIntensity_l2[i],  linestyle='--',
                 marker='o', label = '$\lambda2 = $' + str(l2[i]))

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Default Probability ($)')
    plt.title('Default Probability with Different $\lambda_2$')
    plt.savefig("Proj6_2b.jpg")
    plt.show()

    lam2 = 0.4
    expectedStopTime_l1 = []
    for lam1 in  l1:
            for T in Time:
                expectedStopTime_l1.append(Proj6_2func(lam1, lam2, T)[2])
    expectedStopTime_l1 = np.array(expectedStopTime_l1).reshape((len(l1), len(Time)))
    lam1 = 0.2
    expectedStopTime_l2 = []
    for lam2 in  l2:
            for T in Time:
                expectedStopTime_l2.append(Proj6_2func(lam1, lam2, T)[2])
    expectedStopTime_l2 = np.array(expectedStopTime_l2).reshape((len(l2), len(Time)))

    plt.figure(figsize=(17, 7))
    plt.subplot(1,2,1)
    for i in range(len(expectedStopTime_l1)):
        plt.plot(Time, expectedStopTime_l1[i],  linestyle='--',
                 marker='o', label = '$\lambda1 = $' + str(l1[i]))

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Expected Exercise Time ($\tau$)')
    plt.title('Expected Exercise Time with Different $\lambda_1$')

    plt.subplot(1,2,2)

    for i in range(len(expectedStopTime_l2)):
        plt.plot(Time, expectedStopTime_l2[i],  linestyle='--',
                 marker='o', label = '$\lambda2 = $' + str(l2[i]))

    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Expected Exercise Time ($\tau$)')
    plt.title('Expected Exercise Time with Different $\lambda_2$')


    plt.savefig("Proj6_2c.jpg")
    plt.show()
    return(loan)

q2(seed=7)

