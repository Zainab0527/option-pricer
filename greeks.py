import math
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import seaborn as sns


class OptionGreeks:
    """Calculate option Greeks (sensitivities) using analytical formulas."""

    # Each method is defined as a @staticmethod because they do not depend on any instance (object) of the class or stored data (no `self` is used)
    # functions can be called directly using the class name e.g. OptionGreeks.delta_call(S, K, T, r, sigma).
    
    # Delta measures how much the option’s price changes when the underlying stock price changes by £1
    # A call with delta = 0.6 means: if the stock rises £1, the option increases £0.60
    # For a call, delta ≈ probability the option will expire in the money.

    @staticmethod
    def delta_call(S, K, T, r, sigma):
        """Delta: sensitivity to underlying price change."""
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1)
    

    # A put with delta = -0.4 means: if the stock rises £1, the option decreases £0.40
    # For a put, delta ≈ -1 * probability the option will expire in the money.

    @staticmethod
    def delta_put(S, K, T, r, sigma):
        """Delta for put option."""
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1) - 1
    
    # Gamma measures how fast delta changes as the stock price changes
    # A high gamma means delta changes quickly for small stock price movements
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Gamma: rate of change of delta. Same for calls and puts."""
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    

    # Vega measures how much the option price changes when volatility changes by 1%
    # A vega of 0.2 means: if volatility increases by 1%, the option price increases by £0.20

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Vega: sensitivity to volatility. Per 1% change in volatility."""
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return S * norm.pdf(d1) * math.sqrt(T) / 100
    
    # Theta measures how much the option price changes as time passes, all else equal
    # A theta of -0.05 means: each day that passes, the option price decreases by £0.05
    # Call options typically have negative theta because time value decays each day —
    # as expiry approaches, there’s less time for the stock price to rise above the strike,
    # reducing the option’s overall value.

    @staticmethod
    def theta_call(S, K, T, r, sigma):
        """Theta: time decay for call option. Per day."""
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        return theta
    
    @staticmethod
    def theta_put(S, K, T, r, sigma):
        """Theta: time decay for put option. Per day."""
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        return theta
    
    # How much the option’s price changes for a 1% increase in the risk-free rate
    # A rho of 0.03 means: if interest rates rise by 1%, the option price increases by £0.03

    @staticmethod
    def rho_call(S, K, T, r, sigma):
        """Rho: sensitivity to interest rate. Per 1% change."""
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    
    @staticmethod
    def rho_put(S, K, T, r, sigma):
        """Rho for put option."""
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    
if __name__ == "__main__":
    S = 100     # Stock price
    K = 100     # Strike price
    T = 1       # Time to maturity (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)

    print(f"Call Delta: {OptionGreeks.delta_call(S, K, T, r, sigma):.4f}")
    print(f"Put Delta: {OptionGreeks.delta_put(S, K, T, r, sigma):.4f}")
    print(f"Gamma: {OptionGreeks.gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega: {OptionGreeks.vega(S, K, T, r, sigma):.4f}")
    print(f"Call Theta: {OptionGreeks.theta_call(S, K, T, r, sigma):.4f}")
    print(f"Put Theta: {OptionGreeks.theta_put(S, K, T, r, sigma):.4f}")
    print(f"Call Rho: {OptionGreeks.rho_call(S, K, T, r, sigma):.4f}")
    print(f"Put Rho: {OptionGreeks.rho_put(S, K, T, r, sigma):.4f}")

    