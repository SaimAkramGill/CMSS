import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import time

class CascadeModel:
    """
    Implements the Duncan Watts threshold model of cascades on random networks
    """
    
    def __init__(self, n=10000, z=4, phi_star=0.18, threshold_type="uniform", threshold_std=0, 
                 network_type="uniform", power_law_alpha=2.5):
        """
        Initialize the cascade model with given parameters
        
        Parameters:
        -----------
        n : int
            Number of nodes in the network
        z : float
            Average degree of the network
        phi_star : float
            Mean threshold value
        threshold_type : str
            "uniform" - all nodes have same threshold
            "normal" - thresholds drawn from normal distribution with mean phi_star and std threshold_std
        threshold_std : float
            Standard deviation of threshold distribution (if threshold_type="normal")
        network_type : str
            "uniform" - Erdos-Renyi random graph
            "power_law" - Power-law degree distribution with exponent alpha
        power_law_alpha : float
            Exponent for power-law degree distribution
        """
        self.n = n
        self.z = z
        self.phi_star = phi_star
        self.threshold_type = threshold_type
        self.threshold_std = threshold_std
        self.network_type = network_type
        self.power_law_alpha = power_law_alpha
        
        # Initialize network and thresholds
        self.G = self._create_network()
        self.thresholds = self._assign_thresholds()
        
        # State will be initialized when running simulations
        self.state = None
        
    def _create_network(self):
        """Create network according to specified parameters"""
        if self.network_type == "uniform":
            # Erdos-Renyi random graph with expected average degree z
            p = self.z / (self.n - 1)
            G = nx.erdos_renyi_graph(self.n, p)
            return G
        
        elif self.network_type == "power_law":
            # Create a power-law degree sequence
            # We'll use the configuration model with a power-law degree sequence
            alpha = self.power_law_alpha
            
            # First, generate a sequence that follows power-law with exp cutoff
            # We'll adjust kappa (cutoff) to get the desired average degree z
            
            # Method 1: Use a continuous power law and discretize
            # Here we use an approximate method to create a power-law degree sequence
            # with the correct average degree z
            k_min = 1  # Minimum degree
            
            # For a power law, the average degree is approximately:
            # z ≈ k_min * (alpha-1)/(alpha-2) for alpha > 2
            # We can solve for k_min given z and alpha
            
            # But for simplicity here, we'll generate a sequence and then scale it
            
            # Generate random values from power law distribution
            sequence = np.random.power(alpha, self.n)
            
            # Scale to get reasonable degree values (minimum 1)
            sequence = 1 + (sequence * 10).astype(int)
            
            # Scale to match desired average degree
            current_avg = np.mean(sequence)
            sequence = np.ceil(sequence * (self.z / current_avg)).astype(int)
            
            # Ensure even sum for configuration model
            if sum(sequence) % 2 == 1:
                sequence[0] += 1
                
            # Create graph with this degree sequence
            G = nx.configuration_model(sequence.tolist())
            
            # Remove self-loops and parallel edges
            G = nx.Graph(G)
            return G
        
    def _assign_thresholds(self):
        """Assign thresholds to each node"""
        thresholds = {}
        
        if self.threshold_type == "uniform":
            # All nodes have the same threshold
            for node in self.G.nodes():
                thresholds[node] = self.phi_star
                
        elif self.threshold_type == "normal":
            # Thresholds drawn from normal distribution, clipped to [0,1]
            for node in self.G.nodes():
                # Draw from normal distribution, but clip to [0,1]
                threshold = np.random.normal(self.phi_star, self.threshold_std)
                thresholds[node] = max(0, min(1, threshold))
                
        return thresholds
    
    def initialize_state(self, seed_nodes=None):
        """Initialize all nodes to state 0, except seed nodes set to 1"""
        self.state = {node: 0 for node in self.G.nodes()}
        
        if seed_nodes is None:
            # Select one random node as seed
            seed_nodes = [random.choice(list(self.G.nodes()))]
            
        for node in seed_nodes:
            self.state[node] = 1
            
        return seed_nodes
    
    def run_cascade(self, seed_nodes=None, max_iterations=100):
        """
        Run a cascade simulation from initial seed nodes
        
        Parameters:
        -----------
        seed_nodes : list
            Nodes to activate initially (if None, a random node is chosen)
        max_iterations : int
            Maximum number of iterations to prevent infinite loops
            
        Returns:
        --------
        final_state : dict
            Final state of each node (0 or 1)
        cascade_size : float
            Fraction of nodes activated
        time_steps : int
            Number of iterations until convergence
        """
        # Initialize states
        seed_nodes = self.initialize_state(seed_nodes)
        
        # Track active nodes in each round
        active_this_round = set(seed_nodes)
        time_steps = 0
        
        # Continue until no new activations or max iterations reached
        while active_this_round and time_steps < max_iterations:
            new_active = set()
            
            # Check each node in random order
            nodes_to_check = list(self.G.nodes())
            random.shuffle(nodes_to_check)
            
            for node in nodes_to_check:
                # Skip already active nodes
                if self.state[node] == 1:
                    continue
                    
                # Get neighbors
                neighbors = list(self.G.neighbors(node))
                if not neighbors:
                    continue  # Skip isolated nodes
                    
                # Calculate fraction of active neighbors
                active_neighbors = sum(self.state[neigh] == 1 for neigh in neighbors)
                fraction_active = active_neighbors / len(neighbors)
                
                # Apply threshold rule
                if fraction_active >= self.thresholds[node]:
                    self.state[node] = 1
                    new_active.add(node)
            
            # Update for next round
            active_this_round = new_active
            time_steps += 1
        
        # Calculate cascade size
        cascade_size = sum(self.state.values()) / self.n
        
        return self.state, cascade_size, time_steps
    
    def identify_vulnerable_nodes(self):
        """
        Identify vulnerable nodes (those with threshold <= 1/k)
        
        Returns:
        --------
        vulnerable : set
            Set of vulnerable node IDs
        vulnerable_fraction : float
            Fraction of vulnerable nodes
        """
        vulnerable = set()
        
        for node in self.G.nodes():
            degree = self.G.degree(node)
            if degree == 0:
                continue  # Skip isolated nodes
                
            # A node is vulnerable if its threshold is <= 1/degree
            if self.thresholds[node] <= 1.0 / degree:
                vulnerable.add(node)
                
        return vulnerable, len(vulnerable) / self.n
    
    def run_multiple_simulations(self, num_sims=1000):
        """
        Run multiple cascade simulations with random initial seeds
        
        Parameters:
        -----------
        num_sims : int
            Number of simulations to run
            
        Returns:
        --------
        cascade_sizes : list
            List of cascade sizes as fraction of network
        times_to_converge : list
            List of time steps until convergence
        """
        start_time = time.time()
        
        cascade_sizes = []
        times_to_converge = []
        
        for i in range(num_sims):
            if i > 0 and i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Completed {i} simulations. Elapsed time: {elapsed:.2f}s")
                
            # Select random seed node
            seed = [random.choice(list(self.G.nodes()))]
            
            # Run cascade
            _, size, steps = self.run_cascade(seed)
            
            cascade_sizes.append(size)
            times_to_converge.append(steps)
            
        return cascade_sizes, times_to_converge
    
    def compute_extended_vulnerable_cluster(self, vulnerable_nodes=None):
        """
        Compute the extended vulnerable cluster (vulnerable nodes + their neighbors)
        
        Returns:
        --------
        extended_vulnerable : set
            Set of nodes in the extended vulnerable cluster
        size : float
            Fraction of nodes in the extended vulnerable cluster
        """
        if vulnerable_nodes is None:
            vulnerable_nodes, _ = self.identify_vulnerable_nodes()
            
        extended_vulnerable = set(vulnerable_nodes)
        
        # Add immediate neighbors of vulnerable nodes
        for node in vulnerable_nodes:
            extended_vulnerable.update(self.G.neighbors(node))
            
        return extended_vulnerable, len(extended_vulnerable) / self.n
        
    def compute_analytical_vulnerable_cluster_size(self):
        """
        Compute the fraction of nodes in the vulnerable cluster using the analytical approach
        (This is a simplified version that works for uniform random graphs with homogeneous thresholds)
        
        Returns:
        --------
        Sv : float
            Vulnerable cluster size as fraction of network
        """
        if self.network_type != "uniform" or self.threshold_type != "uniform":
            print("Warning: Analytical solution is approximate for non-uniform graphs or thresholds")
        
        # For homogeneous thresholds, vulnerable nodes are those with degree k such that phi* <= 1/k
        # The max vulnerable degree is floor(1/phi*)
        K_star = int(1.0 / self.phi_star)
        
        # In a Poisson distribution, probability of having degree k is e^(-z) * z^k / k!
        # So the fraction of vulnerable nodes is sum from k=1 to K_star of this probability
        vulnerable_fraction = 0
        for k in range(1, K_star + 1):
            vulnerable_fraction += (np.exp(-self.z) * (self.z ** k)) / np.math.factorial(k)
            
        return vulnerable_fraction