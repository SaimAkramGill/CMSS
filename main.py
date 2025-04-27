#!/usr/bin/env python3
"""
Main script for cascade analysis based on Duncan Watts' Threshold Model
This script coordinates the analysis of cascade dynamics on random networks.
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from cascade_model import CascadeModel

# Import figure generators separately
from figure_generators import generate_figure3, generate_figure4

# Import analysis functions - make sure we use the correct module name
from Cascade_Analysis import (
    calculate_analytical_cascade_condition,
    find_analytical_cascade_window,
    compare_theoretical_vs_simulated,
    verify_critical_points,
    analyze_vulnerable_nodes_distribution,
    analyze_cascade_windows,
    run_full_analysis
)

def main():
    """Main function to coordinate cascade analysis based on command-line arguments"""
    parser = argparse.ArgumentParser(description='Analyze cascade phenomena in networks based on Watts\' threshold model')
    
    # Add arguments for different analysis modes
    parser.add_argument('--full', action='store_true', help='Run the full comprehensive analysis')
    parser.add_argument('--verify-critical', action='store_true', help='Verify critical points')
    parser.add_argument('--cascade-window', action='store_true', help='Generate analytical cascade window')
    parser.add_argument('--vulnerable-nodes', action='store_true', help='Analyze vulnerable nodes distribution')
    parser.add_argument('--figure3', action='store_true', help='Generate Figure 3 (cascade size distributions)')
    parser.add_argument('--figure4', action='store_true', help='Generate Figure 4 (heterogeneity effects)')
    
    # Add parameters that can be modified
    parser.add_argument('--network-size', type=int, default=10000, help='Size of the network')
    parser.add_argument('--num-sims', type=int, default=1000, help='Number of simulations to run')
    parser.add_argument('--phi-star', type=float, default=0.18, help='Threshold value')
    parser.add_argument('--z-value', type=float, default=4.0, help='Average degree')
    parser.add_argument('--output-dir', type=str, default='figures', help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Count how many analysis modes were specified
    analysis_modes = [args.full, args.verify_critical, args.cascade_window, 
                     args.vulnerable_nodes, args.figure3, args.figure4]
    
    # If no specific mode selected, run full analysis
    if not any(analysis_modes):
        print("No specific analysis mode selected, running full analysis...")
        args.full = True
    
    # Execute selected analyses
    if args.full:
        print("Running full comprehensive analysis...")
        run_full_analysis()
    else:
        if args.verify_critical:
            print("\n==== Verifying Critical Points ====")
            verify_critical_points()
            
        if args.cascade_window:
            print("\n==== Generating Analytical Cascade Window ====")
            compare_theoretical_vs_simulated()
            
        if args.vulnerable_nodes:
            print("\n==== Analyzing Vulnerable Nodes Distribution ====")
            analyze_vulnerable_nodes_distribution(
                n=args.network_size, z=args.z_value, phi_star=args.phi_star
            )
            
        if args.figure3:
            print("\n==== Generating Figure 3: Cascade Size Distributions ====")
            generate_figure3(
                n=args.network_size, num_sims=args.num_sims, 
                save_path=f"{args.output_dir}/figure3.png"
            )
            
        if args.figure4:
            print("\n==== Generating Figure 4: Effects of Heterogeneity ====")
            generate_figure4(
                n=args.network_size, num_sims=min(50, args.num_sims),  # Limit for feasibility
                phi_range=np.arange(0.1, 0.3, 0.02), 
                z_range=np.arange(1, 15, 1),
                save_path=f"{args.output_dir}/figure4.png"
            )
    
    print("\nAnalysis complete. Results saved to:", args.output_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")