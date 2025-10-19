import math
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import seaborn as sns

from black_scholes import black_scholes_call

from black_scholes import black_scholes_put

from monte_carlo import monte_carlo_call
        
from monte_carlo import monte_carlo_put


from greeks import OptionGreeks

from implied_volatility import ImpliedVolatility

from heatmaps import SensitivityAnalysis

if __name__ == "__main__":

    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    
    print("=" * 60)
    print("ADVANCED OPTION PRICER - BLACK-SCHOLES & MONTE CARLO")
    print("=" * 60)
    
    # Benchmark Black-Scholes vs Monte Carlo (performance test)
    print("\n[1] PRICING & PERFORMANCE BENCHMARK")
    print("-" * 60)
    
    start = time.time()
    bs_call = black_scholes_call(S, K, T, r, sigma)
    bs_time = time.time() - start
    
    start = time.time()
    mc_call = monte_carlo_call(S, K, T, r, sigma, N=100000)
    mc_time = time.time() - start
    
    pricing_error = abs(bs_call - mc_call) / bs_call * 100
    
    print(f"Black-Scholes Call Price: £{bs_call:.4f} ({bs_time*1000:.2f}ms)")
    print(f"Monte Carlo Call Price:   £{mc_call:.4f} ({mc_time*1000:.2f}ms)")
    print(f"Pricing Accuracy: {pricing_error:.4f}%")
    print(f"Performance: Sub-100ms ✓" if bs_time * 1000 < 100 else f"Performance: {bs_time*1000:.2f}ms")
    
   # Greeks calculation
    print("\n[2] GREEKS CALCULATION (Risk Sensitivities)")
    print("-" * 60)
    
    delta_c = OptionGreeks.delta_call(S, K, T, r, sigma)
    gamma = OptionGreeks.gamma(S, K, T, r, sigma)
    vega = OptionGreeks.vega(S, K, T, r, sigma)
    theta = OptionGreeks.theta_call(S, K, T, r, sigma)
    rho = OptionGreeks.rho_call(S, K, T, r, sigma)
    
    print(f"Delta: {delta_c:.4f} (Δ price per $1 underlying move)")
    print(f"Gamma: {gamma:.4f} (Δ delta per $1 underlying move)")
    print(f"Vega:  {vega:.4f} (Δ price per 1% volatility change)")
    print(f"Theta: {theta:.4f} (Δ price per day)")
    print(f"Rho:   {rho:.4f} (Δ price per 1% interest rate change)")

    #Implied volatility solver
    print("\n[3] IMPLIED VOLATILITY SOLVER")
    print("-" * 60)
    
    market_price = 10.45
    iv = ImpliedVolatility.solve_call(S, K, T, r, market_price)
    recovered_price = black_scholes_call(S, K, T, r, iv)
    
    print(f"Market Price: £{market_price:.4f}")
    print(f"Implied Volatility: {iv:.4f} ({iv*100:.2f}%)")
    print(f"Recovered Price: £{recovered_price:.4f}")
    print(f"Solver Accuracy: {abs(market_price - recovered_price):.6f}")

    # Generate heatmaps
    print("\n[4] GENERATING SENSITIVITY HEATMAPS")
    print("-" * 60)
    
    fig = SensitivityAnalysis.plot_greeks_heatmaps(S, K, T, r, sigma, 'call')
    plt.savefig('option_greeks_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Heatmap saved as 'option_greeks_heatmap.png'")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)