#MFE405 Computational Methods in Finance
#Project 5
#Author: Heyu Zhang


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
def laguerre_polynomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
    x1 = np.exp(-S/2)
    x2 = np.exp(-S/2) * (1 - S)
    x3 = np.exp(-S/2) * (1 - 2*S + S**2/2)
    x4 = np.exp(-S/2) * (1 - 3*S + 3* S**2/2 - S**3/6)
    
    X  = [np.stack([x1, x2], axis = 1),
          np.stack([x1, x2, x3], axis = 1),
          np.stack([x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]
def hermite_polynomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
    x1 = np.ones(S.shape)
    x2 = 2*S
    x3 = 4*S**2 - 2
    x4 = 8*S**3 - 12
    
    X  = [np.stack([x1, x2], axis = 1),
          np.stack([x1, x2, x3], axis = 1),
          np.stack([x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]
def monomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
    x1 = np.ones(S.shape)
    x2 = S
    x3 = S**2
    x4 = S**3 
    
    X  = [np.stack([x1, x2], axis = 1),
          np.stack([x1, x2, x3], axis = 1),
          np.stack([x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]
def LSMC(S0, K, r, sd, T, paths, k, polynomials):
    
    steps =int(np.sqrt(paths)*T)
    St = StockPrices(S0, r, sd, T, paths, steps)/K
    dt = T/steps

    # initialize cash flow matrix
    cashFlow = np.zeros((paths, steps))
    cashFlow[:,steps - 1] = np.maximum(1 - St[:,steps - 1], 0)

    # initialize continuation value matrix
    cont_value = cashFlow

    # initialize stopping time matrix
    decision = np.zeros((paths, steps))
    decision[:, steps - 1] = 1

    # build discount factor
    discountFactor = np.tile(np.exp(-r*dt* np.arange(1, 
                                    steps + 1, 1)), paths).reshape((paths, steps))

    for i in reversed(range(steps - 1)):

        # Find in the money paths
        in_the_money_n = np.where(1 - St[:, i] > 0)[0]
        out_of_money_n = np.asarray(list(set(np.arange(paths)) 
                                            - set(in_the_money_n)))

        #  Use the first k terms of Laguerre Polynomials
        if polynomials == 'laguerre':
            X = laguerre_polynomials(St[in_the_money_n, i], k)
            
        elif polynomials == 'hermite':
            X = hermite_polynomials(St[in_the_money_n, i], k)
            
        elif polynomials == 'monomials':
            X = monomials(St[in_the_money_n, i], k)
        else:
            print ('Error: Please Choose the Right Polynomials to Estimate')
            
            
        Y = cashFlow[in_the_money_n, i + 1]/np.exp(r*dt)

        # Find Least Square Beta
        A = np.dot(X.T, X)
        b = np.dot(X.T, Y)
        Beta = np.dot(np.linalg.pinv(A), b)

        # find continuation value
        cont_value[in_the_money_n,i] =  np.dot(X, Beta)
        try:
            cont_value[out_of_money_n,i] =  cont_value[out_of_money_n, i + 1]/np.exp(r*dt)
        except:
            pass

        # update decision rule
        decision[:, i] = np.where(np.maximum(1 - St[:, i], 0)  - cont_value[:,i] >= 0, 1, 0)
        
        # update cash flow matrix
        cashFlow[:, i] =  np.maximum(1 - St[:, i], cont_value[:,i])
    
    # Find the first occurence of 1, indicating the early exercise date
    first_exercise = np.argmax(decision, axis = 1) 
    decision = np.zeros((len(first_exercise), steps))
    decision[np.arange(len(first_exercise)), first_exercise] = 1
    
    option_value = np.mean(np.sum(decision*discountFactor*cashFlow*K, axis = 1))
    
    return option_value




def q1(S0, X, T, r, sd, N):
    np.random.seed(9)
    k = [2, 3, 4]
    American_put_values_laguerre = {}
    American_put_values_hermite = {}
    American_put_values_monomials = {}
    for t in T:
        for i in k:
            American_put_values_laguerre[(t, i)] = round(LSMC(S0, X, r, sd, t, N, i, 'laguerre'), 5)
            American_put_values_hermite[(t, i)] = round(LSMC(S0, X, r, sd, t, N, i, 'hermite'), 5)
            American_put_values_monomials[(t, i)] = round(LSMC(S0, X, r, sd, t, N, i, 'monomials'), 5)
    columns = ['Maturity T', 'Num terms (k)', 'Value Laguerre ($)']
    American_put_laguerre = pd.concat([pd.DataFrame(list(American_put_values_laguerre.keys())), 
                pd.DataFrame(list(American_put_values_laguerre.values()))], axis  = 1)
    American_put_laguerre.columns = columns
    print(American_put_laguerre)
    American_put_laguerre.index=American_put_laguerre['Maturity T']
    
    columns = ['Maturity T', 'Num terms (k)', 'Value Hermite ($)']
    American_put_hermite = pd.concat([pd.DataFrame(list(American_put_values_hermite.keys())), 
                pd.DataFrame(list(American_put_values_hermite.values()))], axis  = 1)
    American_put_hermite.columns = columns
    print(American_put_hermite)
    American_put_hermite.index=American_put_hermite['Maturity T']
    
    columns = ['Maturity T', 'Num terms (k)', 'Value Monomial ($)']
    American_put_monomials = pd.concat([pd.DataFrame(list(American_put_values_monomials.keys())), 
                pd.DataFrame(list(American_put_values_monomials.values()))], axis  = 1)
    American_put_monomials.columns = columns
    print(American_put_monomials)
    
    #Draw plots
    American_put_monomials.index=American_put_monomials['Maturity T']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
    American_put_laguerre.groupby('Num terms (k)')['Value Laguerre ($)'].plot(legend=True, title='Price_Laguerre', ax=axes[0])
    American_put_hermite.groupby('Num terms (k)')['Value Hermite ($)'].plot(legend=True, title='Price_Hermite',  ax=axes[1])
    American_put_monomials.groupby('Num terms (k)')['Value Monomial ($)'].plot(legend=True, title='Price_Monomial', ax=axes[2])


sd = 0.2
r = 0.06
N=100000
X = 40
S0 = 40
T = [0.5, 1, 2]
q1(S0, X, T, r, sd, N)




