#MFE405 Computational Methods in Finance
#Project 7
#Author: Heyu Zhang


# python set up
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import norm 
set_printoptions(threshold=float('inf'), linewidth= 200, suppress = True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#question1
S0 = arange(4, 17, 1)
dt =  0.002
sd = 0.2
r = 0.04
K = 10
T = 0.5
dx1 = sd*sqrt(dt)
dx2 = sd*sqrt(3*dt)
dx3 = sd*sqrt(4*dt)
def generate_grid(sd, dx):

    return arange(log(16)+dx, log(4)-dx, -dx)
#a
def Generate_Ps_EFD(dt, sd, r, dx):
    Pu = dt*(sd**2 / (2 * dx**2) + (r - 0.5*sd**2)/(2 * dx))
    Pm = 1 - dt*sd**2/dx**2 - r * dt
    Pd =  dt*(sd**2 / (2 * dx**2) - (r - 0.5*sd**2)/(2 * dx))
    return Pu, Pm, Pd
def A_Exlicit(Pu, Pm, Pd, grid_size):
    
    Pu_mtx = hstack((Pu*identity(grid_size - 2), 
                     zeros((grid_size - 2, 2))))

    Pm_mtx = hstack((zeros((grid_size - 2, 1)), 
                     Pm*identity((grid_size - 2)),
                     zeros((grid_size - 2, 1))))

    Pd_mtx = hstack((zeros((grid_size - 2, 2)), 
                    Pd*identity(grid_size - 2)))
    
    A = Pu_mtx + Pm_mtx + Pd_mtx
    A = vstack((A[0,:], A, A[-1,:]))
    
    return A
def ExplicitFinteDifferencePut(dt, sd, r, dx, S0, T):
    
    # Generate stock grids with the input parameters
    log_stock_gird = generate_grid(sd, dx)
    grid_size =  len(log_stock_gird)
    
    # select the index of the stocks that are closest to the $4 - $16 with $1 increment 
    idx = [abs(log_stock_gird - log(i)).argmin() for i in arange(3, 17, 1)]

    # Generate Pu, Pm, Pd
    Pu, Pm, Pd = Generate_Ps_EFD(dt, sd, r, dx)

    # Backward loop through the entire grid, solve the entire stock grid
    A = A_Exlicit(Pu, Pm, Pd, grid_size)
    B =  hstack((zeros(grid_size - 1), 
         exp(log_stock_gird[ -2]) -  exp(log_stock_gird[- 1])))

    F = maximum(K - exp(log_stock_gird), 0)
    for i in range(int(T/dt)):
            F = A.dot(F) + B
    
    # interporlate the stock prices that does not exsit in the grid
    P = interp(arange(4, 17, 1), exp(log_stock_gird[idx]), F[idx])
    
    return P
#b
def Generate_Ps_IFD(dt, sd, r, dx):
    Pu = -1/2* dt*(sd**2 / dx**2 + (r - 0.5*sd**2)/ dx)
    Pm = 1 + dt*sd**2/dx**2 + r * dt
    Pd =  -1/2 * dt*(sd**2 /dx**2 - (r - 0.5*sd**2)/dx)
    return Pu, Pm, Pd
def A_Imlicit(Pu, Pm, Pd, grid_size):
    
    Pu_mtx = hstack((Pu*identity(grid_size - 2), 
                     zeros((grid_size - 2, 2))))

    Pm_mtx = hstack((zeros((grid_size - 2, 1)), 
                     Pm*identity((grid_size - 2)),
                     zeros((grid_size - 2, 1))))

    Pd_mtx = hstack((zeros((grid_size - 2, 2)), 
                    Pd*identity(grid_size - 2)))
    
    A = Pu_mtx + Pm_mtx + Pd_mtx
    A = vstack((hstack((1, -1, zeros(grid_size - 2))),
                A, 
                hstack((zeros(grid_size - 2), -1, 1))))
    
    return A
def ImplicitFinteDifferencePut(dt, sd, r, dx, S0, T):
    
    # Generate stock grids with the input parameters
    log_stock_gird = generate_grid(sd, dx)
    grid_size =  len(log_stock_gird)
    
    # select the index of the stocks that are closest to the $4 - $16 with $1 increment 
    idx = [abs(log_stock_gird - log(i)).argmin() for i in arange(1, 20, 1)]

    # Generate Pu, Pm, Pd
    Pu, Pm, Pd = Generate_Ps_IFD(dt, sd, r, dx)

    # Backward loop through the entire grid, solve the entire stock grid
    A_inv = linalg.inv(A_Imlicit(Pu, Pm, Pd, grid_size))
    
    # initialize matrix B
    B =  hstack((0, maximum(K - exp(log_stock_gird), 0)[1:-1],
                     exp(log_stock_gird[ -2]) -  exp(log_stock_gird[- 1])))

    for i in range(int(T/dt)):
       
        F = A_inv.dot(B)
        B =  hstack((0, F[1:-1],
                     exp(log_stock_gird[ -2]) -  exp(log_stock_gird[- 1])))
    
    # interporlate the stock prices that does not exsit in the grid
    P = interp(S0, exp(log_stock_gird[idx]), F[idx])
    
    return P
#c
def Generate_Ps_CNFD(dt, sd, r, dx):
    Pu = -1/4* dt*(sd**2 / dx**2 + (r - 0.5*sd**2)/ dx)
    Pm = 1 + dt*sd**2/(2*dx**2) + 0.5* r * dt
    Pd =  -1/4 * dt*(sd**2 /dx**2 - (r - 0.5*sd**2)/dx)
    return Pu, Pm, Pd
def A_CNFD(Pu, Pm, Pd, grid_size):
    
    Pu_mtx = hstack((Pu*identity(grid_size - 2), 
                     zeros((grid_size - 2, 2))))

    Pm_mtx = hstack((zeros((grid_size - 2, 1)), 
                     Pm*identity((grid_size - 2)),
                     zeros((grid_size - 2, 1))))

    Pd_mtx = hstack((zeros((grid_size - 2, 2)), 
                    Pd*identity(grid_size - 2)))
    
    A = Pu_mtx + Pm_mtx + Pd_mtx
    A = vstack((hstack((1, -1, zeros(grid_size - 2))),
                A, 
                hstack((zeros(grid_size - 2), -1, 1))))
    
    return A
def CrankNicolsonFinteDifferencePut(dt, sd, r, dx, S0, T):
    
    # Generate stock grids with the input parameters
    log_stock_gird = generate_grid(sd, dx)
    grid_size =  len(log_stock_gird)
    
    # select the index of the stocks that are closest to the $4 - $16 with $1 increment 
    idx = [abs(log_stock_gird - log(i)).argmin() for i in arange(1, 20, 1)]

    # Generate Pu, Pm, Pd
    Pu, Pm, Pd = Generate_Ps_CNFD(dt, sd, r, dx)

    # Backward loop through the entire grid, solve the entire stock grid
    A_inv = linalg.inv(A_CNFD(Pu, Pm, Pd, grid_size))
    
    # initialize matrix B
    payoff = maximum(K - exp(log_stock_gird), 0)
    Z = -Pu*payoff[:-2] - (Pm - 2) * payoff[1:-1] - Pd*payoff[2:]
    B =  hstack((0, Z,
                     exp(log_stock_gird[ -2]) -  exp(log_stock_gird[- 1])))

    for i in range(int(T/dt)):
       
        F = A_inv.dot(B)
        Zi = -Pu*F[:-2] - (Pm - 2) * F[1:-1] - Pd*F[2:]
        B =  hstack((0, Zi,
                     exp(log_stock_gird[ -2]) -  exp(log_stock_gird[- 1])))
    
    # interporlate the stock prices that does not exsit in the grid
    P = interp(S0, exp(log_stock_gird[idx]), F[idx])
    
    return P
def BlackSholes_put(S0, sd, T, K, r):
    # find d1 and d2
    d1 = (log(S0/K) + (r + 0.5*sd**2)*T)/(sd*sqrt(T))
    d2 = d1 - sd*sqrt(T)

    # find call option price
    P =  K*exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1) 
    
    return (P)
bs_put = [BlackSholes_put(s0, 0.2, 0.5, 10, 0.04) for s0 in S0]
bs_put = pd.DataFrame(bs_put, index= S0)
bs_put.columns =['Black Scholes']
explicit_put_values = [ExplicitFinteDifferencePut(dt, sd, r, dx, S0, T) for dx in [dx1, dx2, dx3]]
explicit_put_values = pd.DataFrame(explicit_put_values, columns= S0).T
explicit_put_values = pd.concat([explicit_put_values, bs_put], axis=1)
explicit_put_values.columns = ['Explicit FD dx1', 'Explicit FD dx2', 'Explicit FD dx3', 'Black Scholes']
explicit_put_values['Error dx1'] = explicit_put_values.apply(lambda x: ((x['Explicit FD dx1'] 
                                                                           - x['Black Scholes'])), axis=1)
                                                                          

explicit_put_values['Error dx2'] = explicit_put_values.apply(lambda x: ((x['Explicit FD dx2'] 
                                                                         - x['Black Scholes'])), axis=1)

explicit_put_values['Error dx3'] = explicit_put_values.apply(lambda x: ((x['Explicit FD dx3'] 
                                                                         - x['Black Scholes'])), axis=1)
explicit_put_values.index.name = 'Stock Price (S0)'
explicit_put_values

implicit_put_values = [ImplicitFinteDifferencePut(dt, sd, r, dx, S0, T) for dx in [dx1, dx2, dx3]]
implicit_put_values = pd.DataFrame(implicit_put_values, columns= S0).T
implicit_put_values = pd.concat([implicit_put_values, bs_put], axis=1)
implicit_put_values.columns = ['Implicit FD dx1', 'Implicit FD dx2', 'Implicit FD dx3', 'Black Scholes']

implicit_put_values['Error dx1'] = implicit_put_values.apply(lambda x: (x['Implicit FD dx1'] 
                                                                           - x['Black Scholes']), axis=1)
                                                                          

implicit_put_values['Error dx2'] = implicit_put_values.apply(lambda x: (x['Implicit FD dx2'] 
                                                                         - x['Black Scholes']), axis=1)

implicit_put_values['Error dx3'] = implicit_put_values.apply(lambda x: (x['Implicit FD dx3'] 
                                                                         - x['Black Scholes']), axis=1)
implicit_put_values.index.name = 'Stock Price (S0)'
implicit_put_values


CN_put_values = [CrankNicolsonFinteDifferencePut(dt, sd, r, dx, S0, T) for dx in [dx1, dx2, dx3]]
CN_put_values = pd.DataFrame(CN_put_values, columns= S0).T
CN_put_values = pd.concat([CN_put_values, bs_put], axis=1)
CN_put_values.columns = ['C-N FD dx1', 'C-N FD dx2', 'C-N FD dx3', 'Black Scholes']

CN_put_values['Error dx1'] = CN_put_values.apply(lambda x: x['C-N FD dx1'] - x['Black Scholes'], axis=1)
                                                                          
CN_put_values['Error dx2'] = CN_put_values.apply(lambda x: x['C-N FD dx2'] - x['Black Scholes'], axis=1)

CN_put_values['Error dx3'] = CN_put_values.apply(lambda x: x['C-N FD dx3'] - x['Black Scholes'], axis=1)
CN_put_values.index.name = 'Stock Price (S0)'
CN_put_values




#question2
def question2a(ds, S0, smethod, style):

    sig = 0.2
    r = 0.04
    dt = 0.002
    K = 10
    T = 0.5
    M = int(T / dt)
    if smethod == 1:
        Smin = S0 - 3 * sig * S0
    if smethod == 2:
        Smin = 0.25
    N = int(np.floor((S0 - Smin) / ds))
    S_M = np.array([S0 - i * ds for i in range(-N, N + 1)])

    A = np.zeros((2 * N + 1, 2 * N + 1))

    pu = dt * ((sig * S_M[0]) ** 2 / (2 * (ds ** 2)) + r * S_M[0] / (2 * ds))
    pm = 1 - dt * (sig * S_M[0] / ds) ** 2 - r * dt
    pd = dt * ((sig * S_M[0]) ** 2 / (2 * (ds ** 2)) - r * S_M[0] / (2 * ds))
    pvector = np.array([pu, pm, pd])
    A[0, 0:3] = pvector

    pu = dt * ((sig * S_M[2 * N]) ** 2 / (2 * (ds ** 2)) + r * S_M[2 * N] / (2 * ds))
    pm = 1 - dt * (sig * S_M[2 * N] / ds) ** 2 - r * dt
    pd = dt * ((sig * S_M[2 * N]) ** 2 / (2 * (ds ** 2)) - r * S_M[2 * N] / (2 * ds))
    pvector = np.array([pu, pm, pd])
    A[2 * N, (2 * N - 2):] = pvector

    for i in range(1, 2 * N):
        pu = dt * ((sig * S_M[i]) ** 2 / (2 * (ds ** 2)) + r * S_M[i] / (2 * ds))
        pm = 1 - dt * (sig * S_M[i] / ds) ** 2 - r * dt
        pd = dt * ((sig * S_M[i]) ** 2 / (2 * (ds ** 2)) - r * S_M[i] / (2 * ds))
        pvector = np.array([pu, pm, pd])
        A[i, (i - 1):(i + 2)] = pvector

    if style == "call":
        F = np.maximum(S_M - K, 0)
        B = np.zeros(2 * N + 1)
        B[0] = S_M[0] - S_M[1]
        payoff = np.maximum(S_M - K, 0)
        for i in range(M):
            F = np.matmul(A, F) + B
            F = np.maximum(F, payoff)
        v0 = F[N]

    if style == "put":
        F = np.maximum(K - S_M, 0)
        B = np.zeros(2 * N + 1)
        B[-1] = S_M[-2] - S_M[-1]
        payoff = np.maximum(K - S_M, 0)
        for i in range(M):
            F = np.matmul(A, F) + B
            F = np.maximum(F, payoff)
        v0 = F[N]

    return v0


S0 = [4+i for i in range(13)]
ds = np.array([0.25, 1, 1.25])

q2a1 = np.zeros((13, 6))
for i in range(len(S0)):
    for j in range(3):
        q2a1[i, j] = question2a(ds=ds[j], smethod=1, S0=S0[i], style="call")
for i in range(len(S0)):
    for j in range(3):
        q2a1[i, j+3] = question2a(ds=ds[j], smethod=1, S0=S0[i], style="put")

q2a1 = pd.DataFrame(q2a1)
q2a1.columns = ["call, ds = 0.25", "call, ds = 1", "call, ds = 1.25", "put, ds = 0.25", "put, ds = 1", "put, ds = 1.25"]
q2a1.index = ["S0=" + str(i) for i in S0]
q2a1.to_csv("q2a.csv")


def question2b(ds, S0, smethod, style):

    sig = 0.2
    r = 0.04
    dt = 0.002
    K = 10
    T = 0.5
    M = int(T / dt)
    if smethod == 1:
        Smin = S0 - 3 * sig * S0
    if smethod == 2:
        Smin = 0.25

    N = int(np.floor((S0 - Smin) / ds))
    S_M = np.array([S0 - i * ds for i in range(-N, N + 1)])

    A = np.zeros((2 * N + 1, 2 * N + 1))
    A[0, 0:2] = np.array([1, -1])
    A[2 * N, (2 * N - 1):] = np.array([1, -1])
    for i in range(1, 2 * N):
        pu = -0.5*dt*((sig*S_M[i]/ds)**2 + r*S_M[i]/ds)
        pm = 1 + dt * (sig*S_M[i] / ds) ** 2 + r * dt
        pd = -0.5*dt*((sig*S_M[i]/ds)**2 - r*S_M[i]/ds)
        pvector = np.array([pu, pm, pd])
        A[i, (i - 1):(i + 2)] = pvector
    A_inv = np.linalg.inv(A)

    if style == "call":
        payoff = np.maximum(S_M - K, 0)
        B = payoff
        B[0] = S_M[0] - S_M[1]
        B[-1] = 0
        F = np.zeros(2 * N + 1)
        for i in range(M):
            F = np.matmul(A_inv, B)
            F = np.maximum(F, payoff)
            B[1:2 * N] = F[1:2 * N]
        v0 = F[N]

    if style == "put":
        payoff = np.maximum(K - S_M, 0)
        B = np.maximum(K - S_M, 0)
        B[0] = 0
        B[-1] = S_M[-2] - S_M[-1]
        F = np.zeros(2 * N + 1)
        for i in range(M):
            F = np.matmul(A_inv, B)
            F = np.maximum(F, payoff)
            B[1:2 * N] = F[1:2 * N]
        v0 = F[N]

    return v0


S0 = [4+i for i in range(13)]
ds = np.array([0.25, 1, 1.25])
q2b1 = np.zeros((13, 6))
for i in range(len(S0)):
    for j in range(3):
        q2b1[i, j] = question2b(ds=ds[j], S0=S0[i], smethod=1, style="call")
for i in range(len(S0)):
    for j in range(3):
        q2b1[i, j+3] = question2b(ds=ds[j], S0=S0[i], smethod=1, style="put")

q2b1 = pd.DataFrame(q2b1)
q2b1.columns = ["call, ds = 0.25", "call, ds = 1", "call, ds = 1.25", "put, ds = 0.25", "put, ds = 1", "put, ds = 1.25"]
q2b1.index = ["S0=" + str(i) for i in S0]
q2b1.to_csv("q2b.csv")


def question2c(ds, S0, smethod, style):

    sig = 0.2
    r = 0.04
    dt = 0.002
    K = 10
    T = 0.5
    M = int(T / dt)
    if smethod == 1:
        Smin = S0 - 3 * sig * S0
    if smethod == 2:
        Smin = 0.25
    N = int(np.floor((S0 - Smin) / ds))
    S_M = np.array([S0 - i * ds for i in range(-N, N + 1)])

    A = np.zeros((2 * N + 1, 2 * N + 1))
    A[0, 0:2] = np.array([1, -1])
    A[2 * N, (2 * N - 1):] = np.array([1, -1])
    for i in range(1, 2 * N):
        pu = -0.25*dt*((sig*S_M[i]/ds)**2 + r*S_M[i]/ds)
        pm = 1 + dt * 0.5*(sig*S_M[i] / ds) ** 2 + 0.5*r * dt
        pd = -0.25*dt*((sig*S_M[i]/ds)**2 - r*S_M[i]/ds)
        pvector = np.array([pu, pm, pd])
        A[i, (i - 1):(i + 2)] = pvector
    A_inv = np.linalg.inv(A)

    if style == "call":
        payoff = np.maximum(S_M - K, 0)
        Z = np.zeros(2 * N + 1)
        Z[0] = S_M[0] - S_M[1]
        Z[-1] = 0
        for i in range(1, 2 * N):
            Z[i] = -A[i, i-1] * payoff[i - 1] - (A[i, i] - 2) * payoff[i] - A[i, i+1] * payoff[i + 1]
        F = np.zeros(2 * N + 1)
        for j in range(M):
            F = np.matmul(A_inv, Z)
            F = np.maximum(F, payoff)
            for i in range(1, 2 * N):
                Z[i] = -A[i, i-1] * F[i - 1] - (A[i, i] - 2) * F[i] - A[i, i+1] * F[i + 1]
        v0 = F[N]

    if style == "put":
        payoff = np.maximum(K - S_M, 0)
        Z = np.zeros(2 * N + 1)
        Z[0] = 0
        Z[-1] = S_M[-2] - S_M[-1]
        for i in range(1, 2 * N):
            Z[i] = -A[i, i - 1] * payoff[i - 1] - (A[i, i] - 2) * payoff[i] - A[i, i + 1] * payoff[i + 1]
        F = np.zeros(2 * N + 1)
        for j in range(M):
            F = np.matmul(A_inv, Z)
            F = np.maximum(F, payoff)
            for i in range(1, 2 * N):
                Z[i] = -A[i, i - 1] * F[i - 1] - (A[i, i] - 2) * F[i] - A[i, i + 1] * F[i + 1]
        v0 = F[N]

    return v0


S0 = [4+i for i in range(13)]
ds = np.array([0.25, 1, 1.25])
q2c1 = np.zeros((13, 6))
for i in range(len(S0)):
    for j in range(3):
        q2c1[i, j] = question2c(ds=ds[j], S0=S0[i], smethod=1, style="call")
for i in range(len(S0)):
    for j in range(3):
        q2c1[i, j+3] = question2c(ds=ds[j], S0=S0[i], smethod=1, style="put")
q2c1 = pd.DataFrame(q2c1)
q2c1.columns = ["call, ds = 0.25", "call, ds = 1", "call, ds = 1.25", "put, ds = 0.25", "put, ds = 1", "put, ds = 1.25"]
q2c1.index = ["S0=" + str(i) for i in S0]
q2c1.to_csv("q2c.csv")

plt.plot(S0, q2a1.iloc[:, 0], marker='x', linestyle='-.', label=methods[0]+" ds=0.25")
plt.plot(S0, q2b1.iloc[:, 0], marker='D', linestyle='--', label=methods[1]+" ds=0.25")
plt.plot(S0, q2c1.iloc[:, 0], marker='*', linestyle='-', label=methods[2]+" ds=0.25")
plt.plot(S0, q2a1.iloc[:, 1], marker='x', linestyle='--', label=methods[0]+" ds=1")
plt.plot(S0, q2b1.iloc[:, 1], marker='*', linestyle='-.', label=methods[1]+" ds=1")
plt.plot(S0, q2c1.iloc[:, 1], marker='D', linestyle='-', label=methods[2]+" ds=1")
plt.plot(S0, q2a1.iloc[:, 2], marker='x', linestyle='-', label=methods[0]+" ds=1.25")
plt.plot(S0, q2b1.iloc[:, 2], marker='D', linestyle='-.', label=methods[1]+" ds=1.25")
plt.plot(S0, q2c1.iloc[:, 2], marker='*', linestyle='--', label=methods[2]+" ds=1.25")
plt.title("Call Option Prices")
plt.xlabel("Initial Stock Price")
plt.ylabel("Call Option Prices")
plt.legend(loc="best")
plt.savefig("Q2Call.png")
plt.clf()
plt.close()

plt.plot(S0, q2a1.iloc[:, 3], marker='x', linestyle='-.', label=methods[0]+" ds=0.25")
plt.plot(S0, q2b1.iloc[:, 3], marker='D', linestyle='--', label=methods[1]+" ds=0.25")
plt.plot(S0, q2c1.iloc[:, 3], marker='*', linestyle='-', label=methods[2]+" ds=0.25")
plt.plot(S0, q2a1.iloc[:, 4], marker='x', linestyle='--', label=methods[0]+" ds=1")
plt.plot(S0, q2b1.iloc[:, 4], marker='*', linestyle='-.', label=methods[1]+" ds=1")
plt.plot(S0, q2c1.iloc[:, 4], marker='D', linestyle='-', label=methods[2]+" ds=1")
plt.plot(S0, q2a1.iloc[:, 5], marker='x', linestyle='-', label=methods[0]+" ds=1.25")
plt.plot(S0, q2b1.iloc[:, 5], marker='D', linestyle='-.', label=methods[1]+" ds=1.25")
plt.plot(S0, q2c1.iloc[:, 5], marker='*', linestyle='--', label=methods[2]+" ds=1.25")
plt.title("Put Option Prices")
plt.xlabel("Initial Stock Price")
plt.ylabel("Put Option Prices")
plt.legend(loc="best")
plt.savefig("Q2Put.png")
plt.clf()
plt.close()

