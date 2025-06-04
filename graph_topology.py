import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, distance_matrix
from typing import Tuple, List, Optional
import networkx as nx

class SensorGraphTopology:
    """
    Constructs graph topology for sensor networks supporting various connection strategies.
    
    Supports:
    - Grid-based adjacency (for regular sensor layouts)
    - Delaunay triangulation (for spatial proximity)
    - Distance-based connections (k-nearest neighbors or radius-based)
    - Custom adjacency matrices
    """
    
    def __init__(self, num_sensors: int = 9):
        self.num_sensors = num_sensors
        self.sensor_positions = None
        self.edge_index = None
        self.edge_weights = None
        
    def set_sensor_positions_3x3_grid(self, spacing: float = 1.0) -> np.ndarray:
        """
        Set sensor positions in a 3x3 grid layout.
        
        Args:
            spacing: Distance between adjacent sensors
            
        Returns:
            positions: (num_sensors, 2) array of (x, y) coordinates
        """
        if self.num_sensors != 9:
            raise ValueError("3x3 grid layout requires exactly 9 sensors")
            
        positions = []
        for i in range(3):
            for j in range(3):
                x = j * spacing
                y = i * spacing
                positions.append([x, y])
        
        self.sensor_positions = np.array(positions)
        return self.sensor_positions
    
    def set_custom_sensor_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Set custom sensor positions.
        
        Args:
            positions: (num_sensors, 2) array of (x, y) coordinates
            
        Returns:
            positions: The input positions array
        """
        if positions.shape[0] != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensor positions, got {positions.shape[0]}")
        if positions.shape[1] != 2:
            raise ValueError("Positions must be 2D coordinates (x, y)")
            
        self.sensor_positions = positions
        return self.sensor_positions
    
    def create_grid_adjacency(self, grid_shape: Tuple[int, int] = (3, 3)) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create adjacency for grid-based sensor layout.
        
        Args:
            grid_shape: (rows, cols) shape of the sensor grid
            
        Returns:
            edge_index: (2, num_edges) tensor of edge connections
            edge_weights: (num_edges,) tensor of edge weights (distances)
        """
        rows, cols = grid_shape
        if rows * cols != self.num_sensors:
            raise ValueError(f"Grid shape {grid_shape} doesn't match {self.num_sensors} sensors")
        
        edges = []
        weights = []
        
        # Helper function to convert 2D grid index to 1D sensor index
        def grid_to_sensor_idx(r, c):
            return r * cols + c
        
        # Create edges for grid connectivity (4-connectivity)
        for r in range(rows):
            for c in range(cols):
                current_idx = grid_to_sensor_idx(r, c)
                
                # Right neighbor
                if c < cols - 1:
                    neighbor_idx = grid_to_sensor_idx(r, c + 1)
                    edges.append([current_idx, neighbor_idx])
                    weights.append(1.0)  # Unit distance for adjacent grid cells
                
                # Bottom neighbor
                if r < rows - 1:
                    neighbor_idx = grid_to_sensor_idx(r + 1, c)
                    edges.append([current_idx, neighbor_idx])
                    weights.append(1.0)
        
        # Convert to undirected graph (add reverse edges)
        undirected_edges = edges + [[e[1], e[0]] for e in edges]
        undirected_weights = weights + weights
        
        self.edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
        self.edge_weights = torch.tensor(undirected_weights, dtype=torch.float)
        
        return self.edge_index, self.edge_weights
    
    def create_delaunay_triangulation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph using Delaunay triangulation of sensor positions.
        
        Returns:
            edge_index: (2, num_edges) tensor of edge connections
            edge_weights: (num_edges,) tensor of edge weights (Euclidean distances)
        """
        if self.sensor_positions is None:
            raise ValueError("Sensor positions must be set before creating Delaunay triangulation")
        
        # Create Delaunay triangulation
        tri = Delaunay(self.sensor_positions)
        
        # Extract edges from triangulation
        edges = set()
        for triangle in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([triangle[i], triangle[j]]))
                    edges.add(edge)
        
        # Convert to lists and calculate distances
        edge_list = list(edges)
        weights = []
        
        for edge in edge_list:
            pos1 = self.sensor_positions[edge[0]]
            pos2 = self.sensor_positions[edge[1]]
            distance = np.linalg.norm(pos1 - pos2)
            weights.append(distance)
        
        # Create undirected edge list
        undirected_edges = edge_list + [(e[1], e[0]) for e in edge_list]
        undirected_weights = weights + weights
        
        self.edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
        self.edge_weights = torch.tensor(undirected_weights, dtype=torch.float)
        
        return self.edge_index, self.edge_weights
    
    def create_knn_graph(self, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create k-nearest neighbor graph based on sensor positions.
        
        Args:
            k: Number of nearest neighbors to connect to each sensor
            
        Returns:
            edge_index: (2, num_edges) tensor of edge connections
            edge_weights: (num_edges,) tensor of edge weights (Euclidean distances)
        """
        if self.sensor_positions is None:
            raise ValueError("Sensor positions must be set before creating k-NN graph")
        
        # Compute distance matrix
        dist_matrix = distance_matrix(self.sensor_positions, self.sensor_positions)
        
        edges = []
        weights = []
        
        for i in range(self.num_sensors):
            # Find k nearest neighbors (excluding self)
            distances = dist_matrix[i]
            neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self (index 0)
            
            for neighbor_idx in neighbor_indices:
                edges.append([i, neighbor_idx])
                weights.append(distances[neighbor_idx])
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_weights = torch.tensor(weights, dtype=torch.float)
        
        return self.edge_index, self.edge_weights
    
    def create_radius_graph(self, radius: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph by connecting sensors within a specified radius.
        
        Args:
            radius: Maximum distance for sensor connections
            
        Returns:
            edge_index: (2, num_edges) tensor of edge connections
            edge_weights: (num_edges,) tensor of edge weights (Euclidean distances)
        """
        if self.sensor_positions is None:
            raise ValueError("Sensor positions must be set before creating radius graph")
        
        # Compute distance matrix
        dist_matrix = distance_matrix(self.sensor_positions, self.sensor_positions)
        
        edges = []
        weights = []
        
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):  # Only consider upper triangle
                distance = dist_matrix[i, j]
                if distance <= radius:
                    edges.append([i, j])
                    weights.append(distance)
        
        # Create undirected graph
        undirected_edges = edges + [[e[1], e[0]] for e in edges]
        undirected_weights = weights + weights
        
        self.edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
        self.edge_weights = torch.tensor(undirected_weights, dtype=torch.float)
        
        return self.edge_index, self.edge_weights
    
    def visualize_graph(self, title: str = "Sensor Graph Topology", 
                       node_features: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize the sensor graph topology.
        
        Args:
            title: Plot title
            node_features: Optional node features for coloring (sensors, features)
            figsize: Figure size
        """
        if self.sensor_positions is None or self.edge_index is None:
            raise ValueError("Both sensor positions and edges must be set before visualization")
        
        plt.figure(figsize=figsize)
        
        # Plot edges
        edge_index_np = self.edge_index.numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[:, i]
            x_coords = [self.sensor_positions[src, 0], self.sensor_positions[dst, 0]]
            y_coords = [self.sensor_positions[src, 1], self.sensor_positions[dst, 1]]
            plt.plot(x_coords, y_coords, 'k-', alpha=0.5, linewidth=1)
        
        # Plot nodes
        if node_features is not None:
            # Color nodes based on number of active features
            feature_counts = np.sum(node_features, axis=1)
            scatter = plt.scatter(self.sensor_positions[:, 0], self.sensor_positions[:, 1], 
                                c=feature_counts, cmap='RdYlBu_r', s=200, 
                                edgecolors='black', linewidth=2)
            plt.colorbar(scatter, label='Number of Active Features')
        else:
            plt.scatter(self.sensor_positions[:, 0], self.sensor_positions[:, 1], 
                       c='lightblue', s=200, edgecolors='black', linewidth=2)
        
        # Add sensor labels
        for i in range(self.num_sensors):
            plt.annotate(f'S{i+1}', 
                        (self.sensor_positions[i, 0], self.sensor_positions[i, 1]),
                        ha='center', va='center', fontweight='bold')
        
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def get_graph_statistics(self) -> dict:
        """
        Compute and return graph statistics.
        
        Returns:
            stats: Dictionary containing graph statistics
        """
        if self.edge_index is None:
            raise ValueError("Edges must be set before computing statistics")
        
        # Convert to NetworkX for analysis
        edge_list = self.edge_index.t().numpy().tolist()
        G = nx.Graph()
        G.add_nodes_from(range(self.num_sensors))
        G.add_edges_from(edge_list)
        
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges() // 2,  # Undirected, so divide by 2
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'is_connected': nx.is_connected(G),
            'clustering_coefficient': nx.average_clustering(G),
            'average_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        }
        
        return stats
    
    def print_graph_info(self):
        """Print formatted graph information."""
        if self.edge_index is None:
            print("No graph topology created yet.")
            return
        
        stats = self.get_graph_statistics()
        
        print("\n" + "="*50)
        print("SENSOR GRAPH TOPOLOGY INFORMATION")
        print("="*50)
        print(f"Number of sensors (nodes): {stats['num_nodes']}")
        print(f"Number of connections (edges): {stats['num_edges']}")
        print(f"Average degree: {stats['average_degree']:.2f}")
        print(f"Graph is connected: {stats['is_connected']}")
        print(f"Clustering coefficient: {stats['clustering_coefficient']:.3f}")
        print(f"Average path length: {stats['average_path_length']:.3f}")
        
        if self.edge_weights is not None:
            weights_np = self.edge_weights.numpy()
            print(f"Edge weight statistics:")
            print(f"  Min distance: {np.min(weights_np):.3f}")
            print(f"  Max distance: {np.max(weights_np):.3f}")
            print(f"  Mean distance: {np.mean(weights_np):.3f}")

if __name__ == "__main__":
    # Example usage
    graph_builder = SensorGraphTopology(num_sensors=9)
    
    # Set up 3x3 grid sensor positions
    positions = graph_builder.set_sensor_positions_3x3_grid(spacing=1.0)
    print("Sensor positions:")
    print(positions)
    
    # Create different graph topologies
    print("\n1. Grid-based adjacency:")
    edge_index, edge_weights = graph_builder.create_grid_adjacency()
    graph_builder.print_graph_info()
    graph_builder.visualize_graph("Grid-based Sensor Network")
    
    print("\n2. Delaunay triangulation:")
    edge_index, edge_weights = graph_builder.create_delaunay_triangulation()
    graph_builder.print_graph_info()
    graph_builder.visualize_graph("Delaunay Triangulation Sensor Network")
    
    print("\n3. k-NN graph (k=3):")
    edge_index, edge_weights = graph_builder.create_knn_graph(k=3)
    graph_builder.print_graph_info()
    graph_builder.visualize_graph("k-NN Sensor Network (k=3)")
    
    print("\n4. Radius graph (r=1.5):")
    edge_index, edge_weights = graph_builder.create_radius_graph(radius=1.5)
    graph_builder.print_graph_info()
    graph_builder.visualize_graph("Radius-based Sensor Network (r=1.5)") 