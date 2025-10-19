import math
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import seaborn as sns

# The Black–Scholes model computes an option price given volatility
# but traders often need to find the implied volatility given a market option price

from black_scholes import black_scholes_call, black_scholes_put

class ImpliedVolatility:
    """Solve for implied volatility from market option price."""
    
    @staticmethod
    def solve_call(S, K, T, r, market_price, initial_guess=0.3):
        """Solve for implied volatility of call option using Brent's method."""
        def objective(sigma):
            return abs(black_scholes_call(S, K, T, r, sigma) - market_price)
        # calculates the absolute difference between:
        # The theoretical price from the Black–Scholes formula using volatility sigma, and
        # The actual market price of the call option
        # The goal is to find the sigma that minimizes this difference
        try:
            result = minimize_scalar(objective, bounds=(0.01, 3.0), method='bounded')
            return result.x
        except:
            return initial_guess
        # This uses scipy.optimize.minimize_scalar, which finds the value of sigma that minimizes the objective function
        # bounds=(0.01, 3.0) means the solver will search between 0.01 and 3.0 (i.e., 1% to 300% volatility)
        # method='bounded' means the solver respects those bounds
        # result.x is the value of sigma (volatility) that best fits the market price
        # If something goes wrong (e.g., bad inputs), it falls back to a default


    @staticmethod
    def solve_put(S, K, T, r, market_price, initial_guess=0.3):
        """Solve for implied volatility of put option."""
        def objective(sigma):
            return abs(black_scholes_put(S, K, T, r, sigma) - market_price)
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 3.0), method='bounded')
            return result.x
        except:
            return initial_guess
        


if __name__ == "__main__":
    S = 100     # Stock price
    K = 100     # Strike price
    T = 1       # Time to maturity (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    market_price_call= 10.45
    market_price_put = 5.57

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)

    implied_vol_call = ImpliedVolatility.solve_call(S, K, T, r, market_price_call)
    implied_vol_put = ImpliedVolatility.solve_put(S, K, T, r, market_price_put)
    print(f"Market Call Price: {market_price_call:.2f}, Implied Volatility: {implied_vol_call:.4f}")
    print(f"Market Put Price: {market_price_put:.2f}, Implied Volatility: {implied_vol_put:.4f}")