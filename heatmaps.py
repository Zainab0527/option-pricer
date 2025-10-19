
import math
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import seaborn as sns

from greeks import OptionGreeks



class SensitivityAnalysis:
    """Generate heatmaps for Greeks across market dimensions."""
    
    @staticmethod
    def generate_heatmap(S_base, K, T, r, sigma, option_type='call', greek='delta'):
        """Generate 2D heatmap for underlying price vs volatility."""
        S_range = np.linspace(S_base * 0.7, S_base * 1.3, 20)
        vol_range = np.linspace(0.05, 0.8, 20)
        heatmap = np.zeros((len(vol_range), len(S_range)))
        
        greek_func = {
            ('call', 'delta'): OptionGreeks.delta_call,
            ('call', 'gamma'): OptionGreeks.gamma,
            ('call', 'vega'): OptionGreeks.vega,
            ('call', 'theta'): OptionGreeks.theta_call,
            ('put', 'delta'): OptionGreeks.delta_put,
            ('put', 'gamma'): OptionGreeks.gamma,
            ('put', 'vega'): OptionGreeks.vega,
            ('put', 'theta'): OptionGreeks.theta_put,
        }
        
        func = greek_func.get((option_type, greek), OptionGreeks.delta_call)
        
        for i, vol in enumerate(vol_range):
            for j, S in enumerate(S_range):
                heatmap[i, j] = func(S, K, T, r, vol)
        
        return S_range, vol_range, heatmap
    
    # returns the ranges for underlying prices and volatilities along with a 2D array of heatmap values
    
    @staticmethod
    def plot_greeks_heatmaps(S, K, T, r, sigma, option_type='call'):
        """Plot 2x2 grid of Greeks heatmaps."""
        greeks = ['delta', 'gamma', 'vega', 'theta']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{option_type.upper()} Option Greeks - Sensitivity Heatmaps', fontsize=16)
        
        for idx, greek in enumerate(greeks):
            ax = axes[idx // 2, idx % 2] 
             # axes is a 2x2 array of subplots from plt.subplots(2,2).
             # we want to map a flat index (0–3) to a row and column

            S_range, vol_range, heatmap = SensitivityAnalysis.generate_heatmap( S, K, T, r, sigma, option_type, greek)
            
            
            im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', 
                          extent=[S_range[0], S_range[-1], vol_range[0], vol_range[-1]],
                          origin='lower')
            # ax.imshow() creates a color-coded image from a 2D array
            # aspect = 'auto'automatically adjusts width/height ratio so the heatmap fits the subplot
            # cmap = 'RdYlGn' creates color map: red, yellow, green for low to high values
            # extent = [S_range[0], S_range[-1], vol_range[0], vol_range[-1]]
            # Scales the axes to real values of S and vol, instead of just row/column indices
            # origin = 'lower' places low volatility at the bottom

            ax.set_xlabel('Underlying Price ($)')
            ax.set_ylabel('Volatility')
            ax.set_title(f'{greek.upper()}')
            plt.colorbar(im, ax=ax)
            # Adds a color legend for the heatmap, so you can see what each color represents.
            # attaches the colorbar to this specific subplot.
        
        plt.tight_layout()
        # Adjusts spacing so titles/labels don't overlap
        return fig
    



if __name__ == "__main__":
    S = 100     # Stock price
    K = 100     # Strike price
    T = 1       # Time to maturity (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)

    fig_call = SensitivityAnalysis.plot_greeks_heatmaps(S, K, T, r, sigma, option_type='call')
    
    plt.show()