import math
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Price a European call option using the Black-Scholes formula.
    
    Parameters:
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to maturity (in years)
    r : float - Risk-free interest rate
    sigma : float - Volatility of the underlying asset
    
    Returns:
    float - Call option price
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Price a European put option using the Black-Scholes formula.
    
    Returns:
    float - Put option price
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


if __name__ == "__main__":
    S = 100     # Stock price
    K = 100     # Strike price
    T = 1       # Time to maturity (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)

    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)

    print(f"Call Option Price: {call:.2f}")
    print(f"Put Option Price: {put:.2f}")