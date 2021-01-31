#MFE405 Computational Methods in Finance
#Project 3
#Author: Heyu Zhang

import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import norm 
import numpy as np

#Question1
def simulate_Xt(X0, t, steps, paths):#simulate X
    dt = t/steps
    Xt = [np.array([X0]*paths)]
    for i in range(steps):
        Xt_current = Xt[-1]
        Xt.append(Xt_current + ((1/5) - (1/2)*Xt_current)*dt + (2/3)*math.sqrt(dt)* np.random.normal(0,1, paths))        
    return Xt[-1] #return  Xt for  differenct paths
def simulate_Yt(Y0, t, steps, paths):#simulate Y
    dt = t/steps
    Yt = [np.array([Y0]*paths)]
    for i in range(steps):
        Yt_current = Yt[-1]
        Yt.append(Yt_current + ((2/(((i+1)*dt) + 1))* Yt_current + (((i+1)*dt)**3 + 1)/3)*dt + (((i+1)*dt) **3 + 1)/3 *math.sqrt(dt) * np.random.normal(0, 1, paths))
    return Yt[-1]#return  Xt for  differenct paths
def q1(X0, Y0):
    np.random.seed(7)
    X2 = simulate_Xt(X0=X0, t = 2, steps = 2000, paths = 1000)# Find all simulated paths of X2
    Y2 = simulate_Yt(Y0=Y0, t = 2, steps = 2000, paths = 1000)# Find all simulated paths of Y2
    Y3 = simulate_Yt(Y0=Y0, t = 3, steps = 3000, paths = 1000)# Find all simulated paths of Y3
    P1 = (Y2 > 5).sum()/1000 #calculate P1
    E1 = sum(np.sign(X2) * np.absolute(X2)**(1./3.))/1000 #calculate e1
    E2 = sum(Y3)/1000 #calculate e2
    fXY = list(map(lambda x2, y2: x2*y2 if x2 > 1 else 0, X2, Y2))  #calculate x2y2I(x2>1)
    E3 = sum(fXY)/1000#calculate e3
    print("P(Y2 > 5) = ", P1)# Find the probability of P(Y2) > 5
    print("e1 = ", round(E1, 4))
    print("e2 = ", round(E2,4))
    print("e3 = ", round(E3,4))
q1(X0=1,Y0=0.75)

#Question2
def simulate_Xt(X0, t, steps, paths):#simulate X
    dt = t/steps
    Xt = [np.array([X0]*paths)]
    for i in range(steps):
        Xt_current = Xt[-1]
        Xt.append(Xt_current + ((1/4)*Xt_current) *dt 
                  + (1/3)*Xt_current * math.sqrt(dt)* np.random.normal(0, 1, paths)
                  - (3/4)*Xt_current * math.sqrt(dt)* np.random.normal(0, 1, paths))     
    return Xt[-1]

def simulate_Yt(t, paths):#simulate Y
    Yt = np.exp(-0.08*t 
             + (1/3)* math.sqrt(t)* np.random.normal(0, 1, paths)
             + (3/4)* math.sqrt(t)* np.random.normal(0, 1, paths))
    return Yt

def q2(X0):
    np.random.seed(7)
    X1 = simulate_Xt(X0=X0, t = 1, steps = 1000, paths = 1000)# Find all simulated paths of X1
    X3 = simulate_Xt(X0=X0, t = 3, steps = 3000, paths = 1000)# Find all simulated paths of X3
    Y1 = simulate_Yt(t = 1, paths = 1000)# Find the level of all simulated paths of Y1
    E1 = sum(np.sign(1+X3) * np.absolute(1+X3)**(1./3.))/1000
    E2 = sum(X1*Y1)/1000
    print("e1 = ", round(E1, 4))
    print("e2 = ", round(E2,4))
q2(X0=1)

#Question3
def q3a(S0, sigma, T, X, r, dt=0.004):#0.004 is the default value of dt
    steps=int(T/dt)
    St = [np.array([S0]*10000)]#simulate 10000 paths of St
    St_minus = [np.array([S0]*10000)]
    Wt = []
    Wt_minus = []
    for i in range(steps):
        Wt.append(np.random.normal(0, 1, 10000))
        St_current = St[-1]
        St.append(St_current+St_current*r*dt 
                  + St_current*sigma*math.sqrt(dt)* Wt[i]) 
    ST = St[-1]#return the last value of St for all the paths
    for i in range(steps):
        Wt_minus.append(-Wt[i])#variance reduction techniques: Antithetic Variates
        St_minus_current = St_minus[-1]
        St_minus.append(St_minus_current+St_minus_current*r*dt 
                  + St_minus_current*sigma*math.sqrt(dt)* Wt[i]) 
    ST_minus = St_minus[-1]
    payoff = np.array(list(map(lambda st: st - X if st - X > 0 else 0, ST)))
    payoff_minus = np.array(list(map(lambda st_minus: st_minus - X if st_minus - X > 0 else 0, ST_minus)))
    payoff = (payoff + payoff_minus)/2#Calculate(x1+x2)/2
    C = m.exp(-r*T) * sum(payoff)/len(payoff)#call option price
    return C
def N(x):#simulate the normal distribution curve
    d1 = 0.0498673470
    d2 = 0.0211410061 
    d3 = 0.0032776263
    d4 = 0.0000380036 
    d5 = 0.0000488906 
    d6 = 0.0000053830    
    if x > 0:
        N = 1 - (1/2)*(1 + d1*x + d2*x**2 + d3*x**3 + d4*x**4 + d5*x**5 + d6*x**6) **(-16) 
    else:
        N =  1 - (1 - (1/2)*(1 + d1*(-x) + d2*(-x)**2 + d3*(-x)**3 + d4*(-x)**4 + d5*(-x)**5 + d6*(-x)**6) **(-16) )    
    return N
def q3b(S0, sigma, T, X, r):
    d1 = (math.log(S0/X) + (r + 0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    C = S0*N(d1) - X*m.exp(-r*T)*N(d2)    #B-S formula
    return (C)
def q3c(S0_range, sigma, T, X, r):
    eps = 0.0001#a basis point change
    delta = [(q3b(S0_range[i] + eps, sigma, T, X, r) -  q3b(S0_range[i], sigma, T, X, r)) / eps for i in list(range(len(S0_range)))]#dC/dS
    gamma = [(q3b(S0_range[i] + eps, sigma, T, X, r) -  2* q3b(S0_range[i], sigma, T, X, r)+ q3b(S0_range[i] - eps, sigma, T, X, r))/ (eps)**2 for i in list(range(len(S0_range)))]#dDelta/dS
    theta = [-(q3b(S0_range[i], sigma, T + eps, X, r)  -  q3b(S0_range[i], sigma, T, X, r)) / eps for i in list(range(len(S0_range)))]#-dC/dT
    vega = [(q3b(S0_range[i], sigma + eps, T, X, r)  -  q3b(S0_range[i], sigma, T, X, r)) / eps for i in list(range(len(S0_range)))]#dC/dsigma
    greeks = pd.DataFrame(columns=['Delta', 'Gamma', 'Theta', 'Vega'])
    greeks['Delta']=delta
    greeks['Gamma']=gamma
    greeks['Theta']=theta
    greeks['Vega']=vega
    for i in range(4):#draw plots
        plt.plot(S0_range, greeks.iloc[:, i])
        plt.title(greeks.columns.values.tolist()[i])
        plt.xlabel("S_0 Initial Stock Price at time 0")
        plt.ylabel(greeks.columns.values.tolist()[i])
        plt.show()
q3a(S0=15, sigma=0.25, T=0.5, X=20, r=0.04, dt=0.004)
q3b(S0=15, sigma=0.25, T=0.5, X=20, r=0.04)
q3c(list(range(15, 26)), sigma=0.25, T=0.5, X=20, r=0.04)

#Question4
def simuW1W2(paths,rho):
    Z1 = np.random.normal(0,1, paths)
    Z2 = np.random.normal(0,1, paths)
    Z = np.array([Z1, Z2])
    cov = np.array([[1, rho],[rho, 1]])#covariance matrix
    L = np.linalg.cholesky(cov)
    W1, W2 = L.dot(Z)   
    return [W1, W2]
def St_FT(t, steps, paths, a, b, V0, S0, r, sd,rho):
    dt = t/steps
    Vt = [np.array([V0]*paths)]
    St = [np.array([S0]*paths)]
    for i in range(1, steps + 1):
        dWt1 = np.sqrt(dt)* simuW1W2(paths,rho)[0]
        dWt2 = np.sqrt(dt)* simuW1W2(paths,rho)[1]
        Vt_current = Vt.pop()
        Vt_current_plus = Vt_current.copy()#V_plus
        Vt_current_plus[Vt_current_plus < 0] = 0
        St_current = St.pop()
        St.append(St_current + r*St_current*dt + np.sqrt(Vt_current_plus)*St_current*dWt1)    #f1=V, f2=V_plus,f3=V_plus
        Vt.append(Vt_current + (a* (b - Vt_current_plus))*dt  + sd*np.sqrt(Vt_current_plus)*dWt2)    
    return St[0]
def St_PT(t, steps, paths, a, b, V0, S0, r, sd,rho):
    dt = t/steps
    Vt = [np.array([V0]*paths)]
    St = [np.array([S0]*paths)]
    for i in range(1, steps + 1):
        dWt1 = np.sqrt(dt)* simuW1W2(paths,rho)[0]
        dWt2 = np.sqrt(dt)* simuW1W2(paths,rho)[1]
        Vt_current = Vt.pop()
        Vt_current_plus = Vt_current.copy()#V_plus
        Vt_current_plus[Vt_current_plus < 0] = 0
        St_current = St.pop()
        St.append(St_current + r*St_current*dt + np.sqrt(Vt_current_plus)*St_current*dWt1)   #f1=V, f2=V,f3=V_plus
        Vt.append(Vt_current + (a* (b - Vt_current))*dt  + sd*np.sqrt(Vt_current_plus)*dWt2)        
    return St[0]
def St_R(t, steps, paths, a, b, V0, S0, r, sd,rho):
    dt = t/steps
    Vt = [np.array([V0]*paths)]
    St = [np.array([S0]*paths)]
    for i in range(1, steps + 1):
        dWt1 = np.sqrt(dt)* simuW1W2(paths,rho)[0]
        dWt2 = np.sqrt(dt)* simuW1W2(paths,rho)[1]
        Vt_current = Vt.pop()
        St_current = St.pop()
        St.append(St_current + r*St_current*dt + np.sqrt(abs(Vt_current))*St_current*dWt1)        
        Vt.append(abs(Vt_current) + (a* (b - abs(Vt_current)))*dt + sd*np.sqrt(abs(Vt_current))*dWt2)        #f1=f2=f3=|V|
    return St[0]
def q4(rho, r, S0, K, T, V0, sigma, a, b):
    np.random.seed(999)
    steps=1000
    paths=1000
    st1 =  St_FT(T, steps, paths, a, b, V0, S0, r, sigma, rho)
    st2 = St_PT(T, steps, paths, a, b, V0, S0, r, sigma, rho)
    st3 = St_R(T, steps, paths, a, b, V0, S0, r, sigma, rho)
    payoffs_1 = st1 - K #calculate payoff
    payoffs_2 = st2 - K
    payoffs_3 = st3 - K   
    payoffs_1[payoffs_1 < 0] = 0
    payoffs_2[payoffs_2 < 0] = 0
    payoffs_3[payoffs_3 < 0] = 0   
    C1 = math.exp(-r*T) * sum(payoffs_1)/len(payoffs_1)
    C2 = math.exp(-r*T) * sum(payoffs_2)/len(payoffs_2)
    C3 = math.exp(-r*T) * sum(payoffs_3)/len(payoffs_3)
    print('Full Truncation Method:',C1)
    print('Partial Truncation Method:',C2)
    print('Reflection Method:',C3)
q4(rho=-0.6,r=0.03,S0=48,K=50,T=5,V0=0.05,sigma=0.42,a=5.8,b=0.0625)

#Question5
def q5a(seed, n):#LGM method
    x=seed
    list_LGM=[]
    for i in range(2*n):   
        x=(7**5*x)%(2**31-1)
        list_LGM.append(x/(2**31-1))
    Uni = list(zip(list_LGM[:n], list_LGM[n:]))
    return Uni
def Halton_1d(n, base):#1-dimension Halton squences
    rand = []
    for i in range(n):
        k, m = 0., 1.
        while i > 0:
            i, a = (i // base, i % base)
            m *= base
            k += a / m
        rand.append(k)
    return rand
def Halton_2d(n, base1, base2):    #2-dimension Halton squences
    x = Halton_1d(n, base1)
    y = Halton_1d(n, base2)    
    halton_2d = list(zip(x,y))
    return halton_2d
def q5b():
    return Halton_2d(100, 2, 7)    #using base(2,7)
def q5c():
    return Halton_2d(100, 2, 4)    #using base(2,4)
def q5d():
    Uni = q5a(999,100)
    HB = Halton_2d(100, 2, 7)
    HC = Halton_2d(100, 2, 4)
    plt.scatter(*zip(*Uni))    #Draw 5(a) result
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Uniform [0,1]*[0,1]")
    plt.show()
    
    plt.scatter(*zip(*HB))     #Draw 5(b) result
    plt.title("2-dimensional Halton sequences, using bases 2 and 7")
    plt.xlabel("base 2")
    plt.ylabel("base 7")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()

    plt.scatter(*zip(*HC))     #Draw 5(c) result
    plt.title(" 2-dimensional Halton sequences, using bases 2 and 4")
    plt.xlabel("base 2")
    plt.ylabel("base 4")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()
def integral(b1, b2):
    x1 = Halton_1d(10000, b1)
    x2 = Halton_1d(10000, b2)
    temp = []
    for i in range(10000):    temp.append(np.exp(-x1[i]*x2[i])*(np.sin(6*np.pi*x1[i])+np.sign(np.cos(2*np.pi*x2[i]))*abs(np.cos(2*np.pi*x2[i]))**(1.0/3)))
    est = np.mean(temp)
    return est
def q5e():
    e1 = integral(2, 4)
    e2 = integral(2, 7)
    e3 = integral(5, 7)
    print("Q5e: The estimates of I using bases (2,4), (2,7), (5,7) are respectively:", e1, e2, e3)
q5d()
q5e()
