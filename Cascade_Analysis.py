import numpy as np
import matplotlib.pyplot as plt
from cascade_model import CascadeModel
import time
import os
import math
from collections import Counter
from figure_generators import generate_figure3, generate_figure4

def calculate_analytical_cascade_condition(phi_star, z):
    """
    Calculate the analytical cascade condition G'0(1) = z
    
    For uniform random graphs with Poisson degree distribution and homogeneous thresholds,
    this simplifies to z * Q(K*+1, z) = 1, where K* = floor(1/phi*)
    
    Parameters:
    -----------
    phi_star : float
        Homogeneous threshold
    z : float
        Average degree
        
    Returns:
    --------
    condition_value : float
        Value of G'0(1) - z, where zero means the cascade condition is met
    K_star : int
        Maximum degree for which nodes are vulnerable
    """
    # Maximum degree for which nodes are vulnerable
    K_star = int(1.0 / phi_star)
    
    # Calculate the incomplete gamma function Q(K*+1, z)
    # For a Poisson distribution, this gives the probability that a node has degree <= K*
    # Q(a,x) = Γ(a,x)/Γ(a) where Γ(a,x) is the upper incomplete gamma function
    
    # Simple approach: directly sum the Poisson PMF up to K*
    vulnerable_fraction = 0
    for k in range(1, K_star + 1):
        vulnerable_fraction += (np.exp(-z) * (z ** k)) / math.factorial(k)
    
    # Calculate k(k-1) weighted average for vulnerable nodes
    weighted_sum = 0
    for k in range(1, K_star + 1):
        prob_k = (np.exp(-z) * (z ** k)) / math.factorial(k)
        weighted_sum += k * (k - 1) * prob_k
    
    # The cascade condition is G'0(1) = z
    G_prime_0_1 = weighted_sum / vulnerable_fraction
    
    return G_prime_0_1 - z, K_star

def find_analytical_cascade_window(phi_range, z_range, tolerance=1e-3):
    """
    Find the analytical cascade window boundaries
    
    Parameters:
    -----------
    phi_range : array
        Range of threshold values to evaluate
    z_range : array
        Range of average degree values to evaluate
    tolerance : float
        Tolerance for considering cascade condition met
        
    Returns:
    --------
    window_mask : 2D array
        Boolean mask where True indicates parameters within the cascade window
    """
    window_mask = np.zeros((len(phi_range), len(z_range)), dtype=bool)
    
    for i, phi in enumerate(phi_range):
        for j, z in enumerate(z_range):
            condition_value, _ = calculate_analytical_cascade_condition(phi, z)
            window_mask[i, j] = abs(condition_value) < tolerance
    
    return window_mask

def compare_theoretical_vs_simulated():
    """
    Compare the theoretical cascade condition with simulation results
    """
    # Parameter ranges
    phi_range = np.arange(0.1, 0.31, 0.01)
    z_range = np.arange(1, 16, 0.5)
    
    # Calculate analytical cascade window
    print("Calculating analytical cascade window...")
    window_mask = find_analytical_cascade_window(phi_range, z_range)
    
    # Plot analytical cascade window
    plt.figure(figsize=(10, 8))
    
    # Convert window mask to boundary contours
    X, Y = np.meshgrid(phi_range, z_range)
    plt.contour(X.T, Y.T, window_mask, levels=[0.5], colors='r', linestyles='-', linewidths=2)
    
    plt.xlabel('Threshold (φ*)', fontsize=14)
    plt.ylabel('Average Degree (z)', fontsize=14)
    plt.title('Analytical Cascade Window', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(1, 15)
    
    # Save the figure
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/analytical_window.png", dpi=300, bbox_inches='tight')
    
    print("Analytical cascade window saved to figures/analytical_window.png")
    return plt.gcf()

def verify_critical_points():
    """
    Verify that the critical points used in Figure 3 satisfy the cascade condition
    """
    phi_star = 0.18
    z_lower = 1.05  # Lower critical point
    z_upper = 6.14  # Upper critical point
    
    condition_lower, K_star_lower = calculate_analytical_cascade_condition(phi_star, z_lower)
    condition_upper, K_star_upper = calculate_analytical_cascade_condition(phi_star, z_upper)
    
    print(f"For φ* = {phi_star}, z = {z_lower}:")
    print(f"  Cascade condition G'0(1) - z = {condition_lower:.6f}")
    print(f"  K* = {K_star_lower} (nodes with degree <= {K_star_lower} are vulnerable)")
    print(f"  {'SATISFIES' if abs(condition_lower) < 1e-3 else 'DOES NOT SATISFY'} cascade condition")
    
    print(f"\nFor φ* = {phi_star}, z = {z_upper}:")
    print(f"  Cascade condition G'0(1) - z = {condition_upper:.6f}")
    print(f"  K* = {K_star_upper} (nodes with degree <= {K_star_upper} are vulnerable)")
    print(f"  {'SATISFIES' if abs(condition_upper) < 1e-3 else 'DOES NOT SATISFY'} cascade condition")

def analyze_vulnerable_nodes_distribution(n=10000, z=4, phi_star=0.18):
    """
    Analyze the distribution of vulnerable nodes and their clustering
    """
    print(f"Analyzing vulnerable nodes for network with n={n}, z={z}, φ*={phi_star}")
    
    # Create model
    model = CascadeModel(n=n, z=z, phi_star=phi_star)
    
    # Identify vulnerable nodes
    vulnerable_nodes, vulnerable_fraction = model.identify_vulnerable_nodes()
    
    # Calculate analytical vulnerable fraction
    K_star = int(1.0 / phi_star)
    analytical_fraction = 0
    for k in range(1, K_star + 1):
        analytical_fraction += (np.exp(-z) * (z ** k)) / math.factorial(k)
    
    print(f"Vulnerable node analysis:")
    print(f"  Identified {len(vulnerable_nodes)} vulnerable nodes ({vulnerable_fraction:.4f} of network)")
    print(f"  Analytical vulnerable fraction: {analytical_fraction:.4f}")
    
    # Get degree distribution of vulnerable nodes
    vulnerable_degrees = [model.G.degree(node) for node in vulnerable_nodes]
    degree_counter = Counter(vulnerable_degrees)
    
    # Plot degree distribution of vulnerable nodes
    plt.figure(figsize=(10, 6))
    max_degree = max(vulnerable_degrees)
    degrees = range(1, max_degree + 1)
    counts = [degree_counter.get(d, 0) for d in degrees]
    
    plt.bar(degrees, counts, alpha=0.7)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Degree Distribution of Vulnerable Nodes (φ*={phi_star}, z={z})', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Calculate extended vulnerable cluster
    extended_vulnerable, extended_fraction = model.compute_extended_vulnerable_cluster(vulnerable_nodes)
    
    print(f"Extended vulnerable cluster:")
    print(f"  Size: {len(extended_vulnerable)} nodes ({extended_fraction:.4f} of network)")
    
    # Save the figure
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/vulnerable_degree_dist_z{z}_phi{phi_star}.png", dpi=300, bbox_inches='tight')
    
    return plt.gcf(), vulnerable_fraction, extended_fraction

def analyze_cascade_windows(phi_range=np.arange(0.1, 0.31, 0.01), z_range=np.arange(1, 16, 0.5)):
    """
    Analyze how cascade window changes with threshold distribution
    """
    # Calculate analytical window (homogeneous thresholds)
    print("Calculating analytical cascade window...")
    analytical_window = find_analytical_cascade_window(phi_range, z_range)
    
    # Set up figure
    plt.figure(figsize=(15, 10))
    
    # Plot analytical window
    X, Y = np.meshgrid(phi_range, z_range)
    plt.contour(X.T, Y.T, analytical_window, levels=[0.5], colors='r', 
                linestyles='-', linewidths=2, label='Homogeneous Thresholds')
    
    # Add key points for reference
    reference_points = [
        (0.18, 1.05, 'Lower Critical Point'),
        (0.18, 6.14, 'Upper Critical Point')
    ]
    
    for phi, z, label in reference_points:
        plt.plot(phi, z, 'ko', markersize=8)
        plt.annotate(label, (phi, z), fontsize=10, 
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Threshold (φ*)', fontsize=14)
    plt.ylabel('Average Degree (z)', fontsize=14)
    plt.title('Cascade Window Analysis', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(1, 15)
    
    # Save the figure
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/cascade_window_analysis.png", dpi=300, bbox_inches='tight')
    
    print("Cascade window analysis saved to figures/cascade_window_analysis.png")
    return plt.gcf()

def run_full_analysis():
    """
    Run a comprehensive analysis including all components
    """
    print("Starting comprehensive cascade model analysis...")
    start_time = time.time()
    
    # Create output directory
    os.makedirs("figures", exist_ok=True)
    
    # Step 1: Verify critical points
    print("\n==== Verifying Critical Points ====")
    verify_critical_points()
    
    # Step 2: Generate analytical cascade window
    print("\n==== Generating Analytical Cascade Window ====")
    compare_theoretical_vs_simulated()
    
    # Step 3: Analyze vulnerable nodes distribution
    print("\n==== Analyzing Vulnerable Nodes Distribution ====")
    analyze_vulnerable_nodes_distribution(n=10000, z=4, phi_star=0.18)
    
    # Step 4: Generate Figure 3 - Cascade size distributions at critical points
    print("\n==== Generating Figure 3: Cascade Size Distributions ====")
    generate_figure3(n=1000, num_sims=500, save_path="figures/figure3.png")
    
    # Step 5: Generate Figure 4 - Effects of heterogeneity (reduced parameters for feasibility)
    print("\n==== Generating Figure 4: Effects of Heterogeneity ====")
    generate_figure4(n=2000, num_sims=20, 
                    phi_range=np.arange(0.1, 0.3, 0.02), 
                    z_range=np.arange(1, 15, 1),
                    save_path="figures/figure4.png")
    
    # Step 6: Run additional cascade window analysis
    print("\n==== Analyzing Cascade Windows ====")
    analyze_cascade_windows()
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis complete. Total elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Run the full analysis or individual components as needed
    run_full_analysis()
    
    # Uncomment specific functions to run them individually
    # verify_critical_points()
    # compare_theoretical_vs_simulated()
    # analyze_vulnerable_nodes_distribution(n=10000, z=4, phi_star=0.18)
    # generate_figure3(n=1000, num_sims=500, save_path="figures/figure3.png")