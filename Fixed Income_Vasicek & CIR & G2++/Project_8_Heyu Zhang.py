#MFE405 Computational Methods in Finance
#Project 8
#Author: Heyu Zhang

# python set up
import matplotlib.pyplot as plt
import math
from numpy import*
import pandas as pd
from scipy.stats import * 
import numpy as np
from scipy.optimize import newton
set_printoptions(threshold=float('inf'), linewidth= 200, suppress = True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
random.seed(9)



#Question1
#a
def Vasicek(r0, T, paths, steps):
    
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05
    
    dt = T/steps 
    rt = zeros((paths, steps + 1))
    rt[:, 0] = r0
    for i in range(1, steps + 1):
        rt[:,i] = rt[:,i-1] + kappa*(r_mean - rt[:,i-1]) *dt + sd* sqrt(dt)*random.normal(0, 1, paths)
    return rt
def zeroCouponBondVasicek(r0, t, T, F):
    
    # intialize parameters
    r0= 0.05 
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05
    
    paths = 1000
    steps = int(365*(T - t))
    dt = (T - t)/steps
    r = Vasicek(r0, (T - t), paths, steps)
    
    P = mean(F * exp(-sum(r, axis = 1)*dt))
    
    return P
print("The price of the zero coupon bond under Vasicek: $", 
      round(zeroCouponBondVasicek(r0 = 0.05, t = 0, T = 0.5, F = 1000), 5))


#b
coupons = [30] * 7 + [1030]
payment_date = arange(0.5, 4.5, 0.5)
def couponBondVasicek(r0, t, coupons, payment_date):
    
    # intialize parameters
    r0= 0.05 
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05
    F = 1000
    
    zero_coupon_bonds = list(map(lambda coupon, coupon_date: zeroCouponBondVasicek(r0, t, T = coupon_date, F = coupon), 
                                 coupons, payment_date))
    P = sum(zero_coupon_bonds)
    
    return P
print("The price of the coupon paying bond under Vasicek : $", couponBondVasicek(r0 = 0.05, t = 0, 
                                                                                 coupons = coupons, 
                                                                                 payment_date = payment_date))
#c
def ZeroCouponBondExplicit(rt, t, T, F):
    
    # intialize parameters
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05
    
    # Find bond price given time t
    B = 1/kappa * (1 - exp(-kappa * (T - t)))
    A = exp(((r_mean - sd**2/(2*kappa**2)) * (B - (T - t)) - sd**2/(4*kappa)*B**2))
    
    return F * A*exp(-B*rt)
def callOptionOnZeroExplicit(t, T, K, F):
    
    # intialize parameters
    r0= 0.05 
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05
    
    paths = 15000
    steps = int(365*(T - t))
    dt = (T - t)/steps
    rt = Vasicek(r0, t, paths, steps)
    
    # Find the bond price
    P_bond = ZeroCouponBondExplicit(rt[:,-1], t, T, F)
    
    # Find option payoff
    payoff = maximum(P_bond - K, 0)
    P_option = mean(payoff * exp(-sum(rt, axis = 1)*dt))
    
    return P_option
print("The price of European Option on zero coupon bond (with explicit formula on zero coupon bond): $", 
      round(callOptionOnZeroExplicit(t = 0.25, T = 0.5, K = 980, F = 1000), 5))

#d
def callOptionOnCouponSimulate(t, K, F, coupons, payment_date):
    
    # intialize parameters, t is option maturity
    r0= 0.05 
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05
    
    paths = 1000
    steps = int(365*(t))
    dt = t/steps
    rt = Vasicek(r0, t, paths, steps)
    
    # Find the bond price
    P_bond = array(list(map(lambda rt: couponBondVasicek(rt, t, coupons, payment_date), rt[:, -1])))
    
    # Find option payoff
    payoff = maximum(P_bond - K, 0)
    P_option = mean(payoff * exp(-sum(rt, axis = 1)*dt))
    
    return P_option
print("The price of European Option on coupon paying bond (with monte carlo simulation): $", 
   round(callOptionOnCouponSimulate(t = 0.25, K = 980, F = 1000, coupons = coupons, payment_date = payment_date), 5))

#e
coupons = array([30] * 7 + [1030])
payment_date = arange(0.5, 4.5, 0.5)
maturity = 0.25
def PI(rt, maturity, payment_date):
    
    # intialize parameters, maturity is option maturity, payment_date is coupon payment date
    r0= 0.05 
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05    
    
    # find A(t, T) and B(t, T) according to formula
    B = (1/kappa) * (1 - exp(-kappa * (payment_date - maturity)))
    A = exp((r_mean - (sd**2)/(2*kappa**2)) * (B - (payment_date - maturity) ) - (sd**2)/(4*kappa) * B**2 )
    
    # find pi
    PI_val = A*exp(-B*rt) 
    
    return PI_val
def f_opt(rt, maturity, payment_date, coupons, K):
    
    # find PI
    PIs = PI (rt, maturity, payment_date)
    value = sum(coupons * PIs) - K
    
    return value
def callOptionOnCouponBS( maturity, K, coupons, payment_date):
    
    # intialize parameters, t is option maturity
    r0= 0.05 
    sd = 0.1 
    kappa = 0.82
    r_mean= 0.05    
    
    # find r*
    r_ = newton(f_opt, 0, args=( maturity, payment_date, coupons, K))

    #find Ki
    Ki = PI(r_, maturity, payment_date)
    
    # find P(t, Ti)
    P_t_Ti = PI(r0, 0, payment_date)
    P_t_T = PI(r0, 0, maturity)
    
    # find d1, d2
    sd_p = sd*( (1 - exp(-kappa * (payment_date - maturity)) ) / kappa) * sqrt((1 - exp(-2*kappa*(maturity)))/(2*kappa) )
    
    d1 = log(P_t_Ti/(Ki*P_t_T))/sd_p + sd_p/2
    d2 = d1 - sd_p
    
    # find option price
    c = P_t_Ti*norm.cdf(d1) - Ki*P_t_T*norm.cdf(d2)
    C = sum(coupons * c)
    
    return C
print("The price of European Option on coupon paying bond under Vasicek(with explicit formula): $", 
     round(callOptionOnCouponBS( maturity, 980, coupons, payment_date), 5))

#Question2
#a
K = 980
T = 0.5 # Option maturity, t in q1
S = 1 # bond maturtiy, T in q1
F = 1000 # Face Value of zero coupon bond
r0 = 0.05 
r_mean = 0.055
sd = 0.12
kappa = 0.92 
random.seed(9)
def CIR(r0, sd, kappa, r_mean, T, paths, steps):
    dt = T/steps 
    rt = zeros((paths, steps + 1))
    rt[:, 0] = r0
    for i in range(1, steps + 1):
        rt[:,i] = rt[:,i-1] + kappa*(r_mean - rt[:,i-1]) *dt + sd* sqrt(rt[:,i-1])* sqrt(dt)*random.normal(0, 1, paths)
    return rt
def zeroCouponBondCIR(r0, r_mean, sd, kappa, S, T, F):
    
    paths = 1000
    steps = int(365*(S - T))
    dt = (S - T)/steps
    r = CIR(r0, sd, kappa, r_mean, (S - T), paths, steps)
    P = mean(F * exp(-sum(r, axis = 1)*dt))
    
    return P
def callOptionOnZeroCIR(r0, sd, kappa, r_mean, T, S, K, F, coupons, payment_date):
    
    paths = 1200
    steps = int(365*(S - T))
    dt = (S - T)/steps
    rt = CIR(r0, sd, kappa, r_mean, T, paths, steps)
    
    # Find the bond price
    P_bond = array([zeroCouponBondCIR(r, r_mean, sd, kappa, S, T, F) for r in rt[:, -1]])
    
    # Find option payoff
    payoff = maximum(P_bond - K, 0)
    P_option = mean(payoff * exp(-sum(rt, axis = 1)*dt))
    
    return P_option
print("The price of European Option on zero coupon bond under CIR (with Monte Carlo Simulations): $",
      round(callOptionOnZeroCIR(r0, sd, kappa, r_mean, T, S, K, F, coupons, payment_date), 5))

#b
K = 980
T = 0.5 # Option maturity, t in q1
S = 1 # bond maturtiy, T in q1
F = 1000 # Face Value of zero coupon bond
r0 = 0.05 
r_mean = 0.055
sd = 0.12
kappa = 0.92 
random.seed(9)
def ZeroCouponBondExplicitCIR(rt, r_mean, sd, kappa, T, t, F):
    
    h1 = sqrt(kappa**2 + 2*sd**2)
    h2 = (kappa + h1)/2
    h3 = (2*kappa*r_mean)/sd**2
    
    A = ((h1 * exp(h2*(T - t)))/(h2 * (exp(h1*(T - t)) - 1) + h1))**h3
    B = (exp(h1*(T - t)) - 1)/(h2 * (exp(h1*(T - t)) - 1) + h1)
    
    return F * A*exp(-B*rt)
def generate_grid(sd, dr):

    return arange(0, 0.1 + dr, dr)
def Generate_Ps_IFD(dt, sd, r_mean, dr, kappa, grid_r):

    Pu = -(1/2)*dt*((sd**2 * grid_r)/(dr**2) + (kappa*(r_mean - grid_r))/(dr))
    Pm = 1 + (sd**2 * dt * grid_r)/(dr**2) + grid_r*dt
    Pd = -(1/2)*dt*((sd**2 * grid_r)/(dr**2) - (kappa*(r_mean - grid_r))/(dr))
    
    return Pu, Pm, Pd
def A_Imlicit(Pu, Pm, Pd, grid_size):
    
    Pd_mtx = hstack((diag(Pd[1:-1]), 
                     zeros((grid_size - 2, 2))))

    Pm_mtx = hstack((zeros((grid_size - 2, 1)), 
                     diag(Pm[1:-1]),
                     zeros((grid_size - 2, 1))))

    Pu_mtx = hstack((zeros((grid_size - 2, 2)), 
                    diag(Pu[1:-1])))
    
    A = Pu_mtx + Pm_mtx + Pd_mtx
    A = vstack((hstack((1, -1, zeros(grid_size - 2))),
                A, 
                hstack((zeros(grid_size - 2), -1, 1))))
    
    return A
def ImplicitFinteDifferenceCall(sd, r0, r_mean, kappa, T, S, K, F):
    
    # find grid parameters
    dt = 0.0001
    dr = 0.001 
    F = 1000
    
    # Generate stock grids with the input parameters
    short_rate_gird = generate_grid(sd, dr)
    grid_size =  len(short_rate_gird)
    
    # find the index that r0 exsits on the grid
    r0_idx = where(short_rate_gird == r0)

    # Generate Pu, Pm, Pd
    Pu, Pm, Pd = Generate_Ps_IFD(dt, sd, r_mean, dr, kappa, short_rate_gird)

    # Backward loop through the entire grid, solve the entire stock grid
    A_inv = linalg.inv(A_Imlicit(Pu, Pm, Pd, grid_size))
    
    # initialize matrix B
    B =  hstack(((ZeroCouponBondExplicitCIR(short_rate_gird[0], r_mean, sd, kappa, S, T, F) 
                 - ZeroCouponBondExplicitCIR(short_rate_gird[1], r_mean, sd, kappa, S, T, F)), 
                 maximum(ZeroCouponBondExplicitCIR(short_rate_gird, r_mean, sd, kappa, S, T, F) - K, 0)[1:-1], 0))

    for i in range(int(T/dt)):
       
        option_value = A_inv.dot(B)
        B =  hstack(((ZeroCouponBondExplicitCIR(short_rate_gird[0], r_mean, sd, kappa, S, T, F) 
                 - ZeroCouponBondExplicitCIR(short_rate_gird[1], r_mean, sd, kappa, S, T, F), option_value[1:-1], 0)))
    
    
    return option_value[r0_idx]
print("The price of European Option on zero coupon bond under CIR (with Implicit Finite Difference Method): $",
      round(ImplicitFinteDifferenceCall(sd, r0, r_mean, kappa, T, S, K, F)[0], 5))

#c
def callOptionCIRExplicit(r0, sd, kappa, r_mean, T, S, K, F):
    
    P_0_T = ZeroCouponBondExplicitCIR(r0, r_mean, sd, kappa, T, 0, F)/F
    P_0_S = ZeroCouponBondExplicitCIR(r0, r_mean, sd, kappa, S, 0, F)/F

    K = K/F

    theta = sqrt(kappa**2 + 2*sd**2)
    phi = (2 * theta)/(sd**2 * (exp(theta*(T - 0)) - 1))
    si = (kappa + theta)/sd**2

    h1 = sqrt(kappa**2 + 2*sd**2)
    h2 = (kappa + h1)/2
    h3 = (2*kappa*r_mean)/sd**2

    A = ((h1 * exp(h2*(S - T)))/(h2 * (exp(h1*(S - T)) - 1) + h1))**h3
    B = (exp(h1*(S - T)) - 1)/(h2 * (exp(h1*(S - T)) - 1) + h1)

    r_star = log(A/K)/B

    C = (P_0_S* ncx2.cdf(2*r_star*(phi + si + B), 
                        (4*kappa*r_mean)/(sd**2), 
                        (2*phi**2 * r0 * exp(theta*(T-0)))/(phi + si + B))
                         - K * P_0_T * ncx2.cdf(2*r_star*(phi + si), 
                                                (4*kappa*r_mean)/(sd**2), 
                                                (2*phi**2 * r0 * exp(theta*(T-0)))/(phi + si)))
    C = F*C
    
    return C
K = 980
T = 0.5 # Option maturity, t in q1
S = 1 # bond maturtiy, T in q1
F = 1000 # Face Value of zero coupon bond
r0 = 0.05 
r_mean = 0.055
sd = 0.12
kappa = 0.92
print("The price of European Option on zero coupon bond under CIR (with Monte Carlo Simulations): $",
      round(callOptionCIRExplicit(r0, sd, kappa, r_mean, T, S, K, F), 5))

#Question3
#a
x0 = 0
y0 = 0
r0 = 0.03
option_maturity = 0.5
bond_maturity = 1
K = 950
random.seed(7)
def bivariate_normals(paths):
    
    corr = 0.7
    Z1 = random.normal(0,1, paths)
    Z2 = random.normal(0,1, paths)
    x = Z1
    y = corr*Z1+sqrt(1-corr**2)*Z2
    
    return x,y
def G2_rt(xt, yt, alpha, beta, sigma, eta, phi, T, paths, steps):
    
    # initialize sequence
    dt = T/steps
    x = zeros((paths,steps+1))
    y = zeros((paths,steps+1))
    r = zeros((paths,steps+1))
    
    x[:,0] = xt
    y[:,0] = yt
    # generate correlated dW1,dW2
    dWt = bivariate_normals(paths*steps)
    dWt1 = dWt[0].reshape((paths,steps))
    dWt2 = dWt[1].reshape((paths,steps))
    
    for i in range(1, steps+1):

        x[:,i] = x[:,i-1] -alpha*x[:,i-1]*dt + sigma*sqrt(dt)*dWt1[:,i-1]
        y[:,i] = y[:,i-1] -beta*y[:,i-1]*dt + eta*sqrt(dt)*dWt2[:,i-1]
        
    r = x + y + phi   
    
    return r,x,y
def zeroCouponBondG2(xt, yt, t, T, F):
    phi = 0.03
    alpha = 0.1
    beta = 0.3
    sigma = 0.03
    eta = 0.08
    
    paths = 1000
    steps = 1000
    dt = (T - t)/steps

    # Genetate r paths
    r = G2_rt(xt, yt, alpha, beta, sigma, eta, phi, T, paths, steps)[0]
    
    # Find zero coupon bond price
    P = F * mean(exp(-sum(r[:,1:], axis = 1)*dt))
    
    return P
def putOptionOnZeroG2(xt, yt, opmat, bmat, F, K):
    # set parameters
    phi = 0.03
    alpha = 0.1
    beta = 0.3
    sigma = 0.03
    eta = 0.08
    
    #set paths and steps 
    paths = 1000
    steps = 1000
    dt = (bmat - opmat)/steps
    
    # all 3 processes from t =0 to T = option_maturity
    r_opmat,x_opmat,y_opmat = G2_rt(xt, yt, alpha, beta, sigma, eta, phi, opmat, paths, steps)
    
    # bond value in 2nd period T-S
    p_bond = array(list(map(lambda x,y: zeroCouponBondG2(x, y, opmat,bmat,F), x_opmat[:,-1], y_opmat[:,-1]))) 
    
    # Find option payoff
    payoff = maximum(K - p_bond, 0)
    P_option = mean(payoff * exp(-sum(r_opmat[:,1:], axis = 1)*dt))
    
    return P_option
print("The zero coupon bond price under two factor models: $",
        putOptionOnZeroG2(x0, y0, opmat= option_maturity, bmat =bond_maturity , F = 1000, K =950))
def zeroCouponBondExplicit(t, T, F):
    
    # set parameters
    phi = 0.03
    alpha = 0.1
    beta = 0.3
    sigma = 0.03
    eta = 0.08
    
    # break down the long fomula into 3 parts
    V1 = (sigma**2/ alpha**2)*( (T - t) + (2/alpha)*exp(-alpha*(T - t)) - (1/(2*alpha)) *exp(-2*alpha*(T - t)) - (3/(2*alpha))) 
    
    V2 = (eta**2/beta**2) *((T - t) + (2/beta)*exp(-beta * (T - t)) - (1/(2*beta)) *exp(-2 * beta * (T - t)) - (3/(2 * beta)))
        
    V3 = 2*0.7*(sigma*eta)/(alpha*beta)*((T - t) + (exp(-alpha*(T - t)) - 1)/alpha 
                                           + (exp(-beta*(T - t)) - 1)/beta -(exp(-(alpha + beta)*(T - t))-1)/(alpha + beta))
    Vt = V1 + V2 + V3
    
    P = exp(-phi*(T - t) - ((1 - exp(-alpha * (T - t)))/alpha) * x0 - ((1 - exp(-beta * (T - t))/beta)*y0) + (1/2)* Vt) * F
    
    return P
def europeanOptionExplicit(t, T, S, K, F):
    
    K = K/F
    xt = 0
    yt = 0
    corr = 0.7
    alpha = 0.1
    beta = 0.3
    sigma = 0.03
    eta = 0.08
    
    # break down the long formula into 3 parts
    sigma1 = sigma**2/(2* alpha**3)*(1 - exp(-alpha * (S - T)))**2 * (1 - exp(-2 *alpha *(T - t)))
    
    sigma2 = eta**2/(2*beta**3)*(1 - exp(-beta* (S - T)))**2*(1 - exp(-2*beta*(T - t)))
    
    sigma3 = (2 *0.7 *sigma *eta/(alpha*beta*(alpha + beta))
              *(1-exp(-alpha*(S - T)))*(1 - exp(-beta*(S - T)))*(1 -exp(-(alpha + beta)*(T - t))))
    
    sigma_sqr = sqrt(sigma1 + sigma2 + sigma3)

    p_t_S = zeroCouponBondExplicit(0, S, F)
    p_t_T = zeroCouponBondExplicit(0, T, F)
    
    put = (-p_t_S*norm.cdf(log(K*p_t_T/p_t_S)/sigma_sqr - sigma_sqr/2) 
           + p_t_T*K*norm.cdf(log(K*p_t_T/p_t_S)/sigma_sqr +sigma_sqr/2))
    return put
print("The zero coupon bond price with explicit formula: $",
     round(europeanOptionExplicit(t=0, T=0.5, S =1, K=950, F=1000), 5))
