import math
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import seaborn as sns



def monte_carlo_call(S, K, T, r, sigma, N=100000, seed=42):
    """
    Price European call using Monte Carlo simulation:
        - simulates thousands of possible future price paths for the underlying asset 
        - and averages the discounted payoffs at maturity

     Parameters:
     S : Current stock price
     K : Strike price
     T : Time to maturity in years
     r : Risk-free interest rate
     sigma : Volatility of the underlying asset
     N : Number of simulations
     """
    
    np.random.seed(seed) 
    # sets the starting point for  random number generator
    # identical random numbers each time you run the simulation, ensuring reproducibility
    
    dt = T / 252         
    # daily time step size assuming 252 trading days in a year
    # gives the length of one trading day in years

    steps = int(T * 252) 
    # total number of time steps in the simulation
    
    paths = np.zeros((N, steps + 1))
    paths[:, 0] = S
    # Generate price paths by creating 2D array of zeroes
    # rows represent N simulated paths
    # columns represent time steps including time 0
    # first column initialised to current stock price S

    
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(N)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    # Simulates the price paths using Geometric Brwonian Motion (GBM model)
    # for each time step< generates N random normal variables Z, one for each path
    # Drift term (r - 0.5 * sigma**2) * dt accounts for average growth at risk-free rate adjusted for volatilty
    # Diffusion term sigma * np.sqrt(dt) * Z adds randomness based on volatility and time step size
    

    
    payoffs = np.maximum(paths[:, -1] - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    return call_price
    # Compute payoff for each simulated path at expiry using max(S_T ​− K, 0)
    # Discounts average payoff back to present value using exp(-r * T)



def monte_carlo_put(S, K, T, r, sigma, N=100000, seed=42):

    np.random.seed(seed)
    dt = T / 252
    steps = int(T * 252)

    paths = np.zeros(( N, steps + 1))
    paths[:, 0] = S

    for t in range(1, steps + 1):
        Z = np.random.standard_normal(N)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    payoffs = np.maximum(K - paths[:, -1], 0)
    put_price = np.exp(-r * T) * np.mean(payoffs)
    return put_price

if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 100000
    seed = 42


    call_mc = monte_carlo_call(S, K, T, r, sigma, N, seed)
    put_mc = monte_carlo_put(S, K, T, r, sigma, N, seed)

    print(f"Monte Carlo Call Price: {call_mc:.2f}")
    print(f"Monte Carlo Put Price: {put_mc:.2f}")



