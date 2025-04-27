import numpy as np
import matplotlib.pyplot as plt
from cascade_model import CascadeModel
import time
import os
from collections import Counter

def generate_figure3(n=1000, num_sims=1000, save_path=None):
    """
    Generate Figure 3: Cascade size distributions at critical points
    
    Parameters:
    -----------
    n : int
        Network size
    num_sims : int
        Number of simulations for each parameter set
    save_path : str
        Path to save the figure (if None, figure is displayed but not saved)
    """
    print("Generating Figure 3: Cascade size distributions at critical points")
    
    # Parameters for the lower and upper critical points
    phi_star = 0.18
    z_lower = 1.05  # Lower critical point
    z_upper = 6.14  # Upper critical point
    
    # Run simulations at lower critical point
    print(f"Running {num_sims} simulations at lower critical point (z={z_lower})...")
    model_lower = CascadeModel(n=n, z=z_lower, phi_star=phi_star)
    cascade_sizes_lower, _ = model_lower.run_multiple_simulations(num_sims=num_sims)
    
    # Run simulations at upper critical point
    print(f"Running {num_sims} simulations at upper critical point (z={z_upper})...")
    model_upper = CascadeModel(n=n, z=z_upper, phi_star=phi_star)
    cascade_sizes_upper, _ = model_upper.run_multiple_simulations(num_sims=num_sims)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Function to compute cumulative distribution
    def compute_ccdf(sizes):
        # Count occurrences of each size
        counter = Counter(sizes)
        # Sort sizes in descending order
        sorted_sizes = sorted(counter.keys(), reverse=True)
        # Compute cumulative counts
        cumulative_counts = []
        count_so_far = 0
        for size in sorted_sizes:
            count_so_far += counter[size]
            cumulative_counts.append(count_so_far)
        
        return sorted_sizes[::-1], cumulative_counts[::-1]  # Reverse to get ascending sizes
    
    # Compute CCDFs
    sizes_lower, counts_lower = compute_ccdf(cascade_sizes_lower)
    sizes_upper, counts_upper = compute_ccdf(cascade_sizes_upper)
    
    # Plot cumulative distributions on log-log scale
    plt.loglog(sizes_lower, counts_lower, 'o', markersize=8, markerfacecolor='none', 
               markeredgecolor='black', markeredgewidth=1.5, label=f'Lower critical point (z={z_lower})')
    
    plt.loglog(sizes_upper, counts_upper, '.', markersize=10, color='black', 
               label=f'Upper critical point (z={z_upper})')
    
    # Add reference line with slope -1/2 for comparison with theoretical value
    # (the slope -3/2 is for the PDF, for CCDF it's -1/2)
    x_range = np.logspace(-3, 0, 100)
    y_range = num_sims * x_range**(-0.5)  # Slope -1/2 for cumulative distribution
    plt.loglog(x_range, y_range, 'k-', linewidth=1, label='Slope -1/2')
    
    plt.xlabel('Cascade Size', fontsize=14)
    plt.ylabel('Cumulative Distribution', fontsize=14)
    plt.title('Cascade Size Distributions at Critical Points', fontsize=16)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return plt.gcf()

def generate_figure4(n=5000, phi_range=np.arange(0.10, 0.31, 0.01), z_range=np.arange(1, 20, 1), 
                    num_sims=50, save_path=None):
    """
    Generate Figure 4: Effects of heterogeneity on cascade windows
    
    Parameters:
    -----------
    n : int
        Network size (smaller than paper for computational feasibility)
    phi_range : array
        Range of threshold values to evaluate
    z_range : array
        Range of average degree values to evaluate
    num_sims : int
        Number of simulations per parameter combination
    save_path : str
        Path to save the figure (if None, figure is displayed but not saved)
    """
    print("Generating Figure 4: Effects of heterogeneity on cascade windows")
    
    # This is a computation-intensive function that tries to recreate the cascade windows
    # We're using a smaller network and fewer simulations than the paper for feasibility
    
    # Create matrices to store cascade frequencies for different parameter combinations
    cascade_freq_uniform = np.zeros((len(phi_range), len(z_range)))
    cascade_freq_normal_005 = np.zeros((len(phi_range), len(z_range)))
    cascade_freq_normal_010 = np.zeros((len(phi_range), len(z_range)))
    cascade_freq_powerlaw = np.zeros((len(phi_range), len(z_range)))
    
    # Function to determine if cascade is global (>10% of network)
    def is_global_cascade(size):
        return size > 0.1
    
    # Go through all parameter combinations
    total_combinations = len(phi_range) * len(z_range)
    counter = 0
    start_time = time.time()
    
    for i, phi in enumerate(phi_range):
        for j, z in enumerate(z_range):
            counter += 1
            print(f"Testing parameters {counter}/{total_combinations}: φ={phi:.2f}, z={z:.1f}")
            print(f"Elapsed time: {time.time() - start_time:.1f}s")
            
            # 1. Uniform graph with uniform thresholds (baseline)
            model_uniform = CascadeModel(n=n, z=z, phi_star=phi, 
                                         threshold_type="uniform", network_type="uniform")
            sizes_uniform, _ = model_uniform.run_multiple_simulations(num_sims=num_sims)
            cascade_freq_uniform[i, j] = sum(is_global_cascade(s) for s in sizes_uniform) / num_sims
            
            # 2. Uniform graph with normal thresholds (σ=0.05)
            model_normal_005 = CascadeModel(n=n, z=z, phi_star=phi, 
                                           threshold_type="normal", threshold_std=0.05, 
                                           network_type="uniform")
            sizes_normal_005, _ = model_normal_005.run_multiple_simulations(num_sims=num_sims)
            cascade_freq_normal_005[i, j] = sum(is_global_cascade(s) for s in sizes_normal_005) / num_sims
            
            # 3. Uniform graph with normal thresholds (σ=0.10)
            model_normal_010 = CascadeModel(n=n, z=z, phi_star=phi, 
                                           threshold_type="normal", threshold_std=0.1, 
                                           network_type="uniform")
            sizes_normal_010, _ = model_normal_010.run_multiple_simulations(num_sims=num_sims)
            cascade_freq_normal_010[i, j] = sum(is_global_cascade(s) for s in sizes_normal_010) / num_sims
            
            # 4. Power-law graph with uniform thresholds
            model_powerlaw = CascadeModel(n=n, z=z, phi_star=phi, 
                                         threshold_type="uniform", network_type="power_law", 
                                         power_law_alpha=2.5)
            sizes_powerlaw, _ = model_powerlaw.run_multiple_simulations(num_sims=num_sims)
            cascade_freq_powerlaw[i, j] = sum(is_global_cascade(s) for s in sizes_powerlaw) / num_sims
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SIMPLIFIED: Directly create the contour plots without trying to label them
    # Panel A - Threshold Heterogeneity
    ax1.contour(phi_range, z_range, cascade_freq_uniform.T, levels=[0.05], colors=['k'], linestyles=['-'])
    ax1.contour(phi_range, z_range, cascade_freq_normal_005.T, levels=[0.05], colors=['b'], linestyles=['--'])
    ax1.contour(phi_range, z_range, cascade_freq_normal_010.T, levels=[0.05], colors=['r'], linestyles=[':'])
    
    # Add a legend for panel A
    from matplotlib.lines import Line2D
    legend_elements_a = [
        Line2D([0], [0], color='k', linestyle='-', lw=2, label='Homogeneous Thresholds'),
        Line2D([0], [0], color='b', linestyle='--', lw=2, label='σ=0.05'),
        Line2D([0], [0], color='r', linestyle=':', lw=2, label='σ=0.1')
    ]
    ax1.legend(handles=legend_elements_a, loc='upper right')
    
    ax1.set_xlabel('Threshold (φ*)', fontsize=12)
    ax1.set_ylabel('Average Degree (z)', fontsize=12)
    ax1.set_title('(a) Effect of Threshold Heterogeneity', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Panel B - Network Heterogeneity
    ax2.contour(phi_range, z_range, cascade_freq_uniform.T, levels=[0.05], colors=['k'], linestyles=['-'])
    ax2.contour(phi_range, z_range, cascade_freq_powerlaw.T, levels=[0.05], colors=['g'], linestyles=['--'])
    
    # Add a legend for panel B
    legend_elements_b = [
        Line2D([0], [0], color='k', linestyle='-', lw=2, label='Uniform Random Graph'),
        Line2D([0], [0], color='g', linestyle='--', lw=2, label='Scale-Free Random Graph')
    ]
    ax2.legend(handles=legend_elements_b, loc='upper right')
    
    ax2.set_xlabel('Threshold (φ*)', fontsize=12)
    ax2.set_ylabel('Average Degree (z)', fontsize=12)
    ax2.set_title('(b) Effect of Network Heterogeneity', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Generate Figure 3
    generate_figure3(n=1000, num_sims=1000, save_path="figures/figure3.png")
    
    # Generate Figure 4 (this will take a long time!)
    generate_figure4(n=2000, num_sims=20, 
                    phi_range=np.arange(0.1, 0.3, 0.02), 
                    z_range=np.arange(1, 15, 1),
                    save_path="figures/figure4.png")