#!/usr/bin/env python3
"""
Demo script for Spiking Graph Neural Networks for Damage Detection

This script demonstrates the basic functionality of each component without
requiring the full training pipeline or actual data files.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from binary_feature_extractor import BinaryFeatureExtractor
from graph_topology import SensorGraphTopology
from spiking_gnn_model import SpikingGNN

def generate_synthetic_strain_data(num_timesteps=1000, num_sensors=9, damage_present=False):
    """
    Generate synthetic strain data for testing.
    
    Args:
        num_timesteps: Number of time steps
        num_sensors: Number of sensors
        damage_present: Whether to simulate damage
        
    Returns:
        strain_data: Synthetic strain time-series (timesteps, sensors)
    """
    # Base strain pattern
    time = np.linspace(0, 10, num_timesteps)
    strain_data = np.zeros((num_timesteps, num_sensors))
    
    for sensor in range(num_sensors):
        # Base oscillation with some random variation
        base_strain = 0.002 * np.sin(2 * np.pi * 0.5 * time) + 0.001 * np.random.randn(num_timesteps)
        
        if damage_present and sensor == 4:  # Center sensor damage
            # Add damage signature: higher amplitude and spikes
            damage_factor = 2.0
            spike_times = np.random.choice(num_timesteps, size=int(0.1 * num_timesteps), replace=False)
            base_strain *= damage_factor
            base_strain[spike_times] += 0.01 * np.random.randn(len(spike_times))
        
        strain_data[:, sensor] = base_strain
    
    return strain_data

def demo_binary_feature_extraction():
    """Demonstrate binary feature extraction."""
    print("="*60)
    print("DEMO 1: BINARY FEATURE EXTRACTION")
    print("="*60)
    
    # Create extractor
    extractor = BinaryFeatureExtractor()
    
    # Generate synthetic data
    intact_strain = generate_synthetic_strain_data(damage_present=False)
    damaged_strain = generate_synthetic_strain_data(damage_present=True)
    
    print(f"Generated strain data:")
    print(f"  Intact shape: {intact_strain.shape}")
    print(f"  Damaged shape: {damaged_strain.shape}")
    
    # Extract binary features
    intact_features = extractor.extract_features_all_sensors(intact_strain)
    damaged_features = extractor.extract_features_all_sensors(damaged_strain)
    
    print(f"\nBinary features extracted:")
    print(f"  Intact features shape: {intact_features.shape}")
    print(f"  Damaged features shape: {damaged_features.shape}")
    
    print(f"\nIntact binary features:")
    print(intact_features)
    print(f"\nDamaged binary features:")
    print(damaged_features)
    
    # Compare features
    stats = extractor.compare_intact_vs_damaged(intact_features, damaged_features)
    extractor.print_comparison_report(stats)
    
    return intact_features, damaged_features

def demo_graph_topology():
    """Demonstrate graph topology construction."""
    print("\n" + "="*60)
    print("DEMO 2: GRAPH TOPOLOGY CONSTRUCTION")
    print("="*60)
    
    # Create graph builder
    graph_builder = SensorGraphTopology(num_sensors=9)
    
    # Set sensor positions
    positions = graph_builder.set_sensor_positions_3x3_grid(spacing=1.0)
    print(f"Sensor positions (3x3 grid):")
    print(positions)
    
    # Test different graph topologies
    topologies = ["grid", "delaunay", "knn", "radius"]
    
    for topo in topologies:
        print(f"\n--- {topo.upper()} TOPOLOGY ---")
        
        if topo == "grid":
            edge_index, edge_weights = graph_builder.create_grid_adjacency()
        elif topo == "delaunay":
            edge_index, edge_weights = graph_builder.create_delaunay_triangulation()
        elif topo == "knn":
            edge_index, edge_weights = graph_builder.create_knn_graph(k=3)
        elif topo == "radius":
            edge_index, edge_weights = graph_builder.create_radius_graph(radius=1.5)
        
        graph_builder.print_graph_info()
    
    return graph_builder

def demo_sgnn_model():
    """Demonstrate SGNN model functionality."""
    print("\n" + "="*60)
    print("DEMO 3: SPIKING GRAPH NEURAL NETWORK")
    print("="*60)
    
    # Model parameters
    num_sensors = 9
    num_features = 5
    
    # Create model
    model = SpikingGNN(
        num_sensors=num_sensors,
        num_features=num_features,
        hidden_dim=32,
        num_layers=2,
        lif_threshold=1.0,
        lif_decay=0.9
    )
    
    print(f"SGNN Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Architecture: {model}")
    
    # Create sample data
    x = torch.randn(num_sensors, num_features)  # Random node features
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7], 
                              [1, 0, 2, 1, 4, 3, 5, 4, 7, 6]], dtype=torch.long)
    
    print(f"\nInput data:")
    print(f"  Node features shape: {x.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x, edge_index)
        
        print(f"\nModel outputs:")
        print(f"  Global logits shape: {outputs['global_logits'].shape}")
        print(f"  Global logits: {outputs['global_logits']}")
        print(f"  Node probabilities shape: {outputs['node_probs'].shape}")
        print(f"  Spike counts per layer: {outputs['spike_counts']}")
        
        # Make prediction
        predictions = model.predict(x, edge_index)
        print(f"\nPredictions:")
        print(f"  Global damage: {'Damaged' if predictions['global_damage'] == 1 else 'Intact'}")
        print(f"  Confidence: {predictions['global_confidence']:.3f}")
        print(f"  Node damage probabilities: {predictions['node_damage_probs']}")
    
    return model

def demo_integrated_pipeline():
    """Demonstrate integrated pipeline with synthetic data."""
    print("\n" + "="*60)
    print("DEMO 4: INTEGRATED PIPELINE")
    print("="*60)
    
    # Generate synthetic data
    intact_strain = generate_synthetic_strain_data(num_timesteps=500, damage_present=False)
    damaged_strain = generate_synthetic_strain_data(num_timesteps=500, damage_present=True)
    
    # Feature extraction
    extractor = BinaryFeatureExtractor()
    intact_features = extractor.extract_features_all_sensors(intact_strain)
    damaged_features = extractor.extract_features_all_sensors(damaged_strain)
    
    # Graph construction
    graph_builder = SensorGraphTopology(num_sensors=9)
    graph_builder.set_sensor_positions_3x3_grid()
    edge_index, edge_weights = graph_builder.create_delaunay_triangulation()
    
    # Model creation
    model = SpikingGNN(
        num_sensors=9,
        num_features=5,
        hidden_dim=32,
        num_layers=2
    )
    
    # Test predictions on both intact and damaged features
    model.eval()
    with torch.no_grad():
        # Intact prediction
        x_intact = torch.tensor(intact_features, dtype=torch.float32)
        pred_intact = model.predict(x_intact, edge_index, edge_weights)
        
        # Damaged prediction
        x_damaged = torch.tensor(damaged_features, dtype=torch.float32)
        pred_damaged = model.predict(x_damaged, edge_index, edge_weights)
        
        print(f"Intact sample prediction:")
        print(f"  Damage: {'Damaged' if pred_intact['global_damage'] == 1 else 'Intact'}")
        print(f"  Confidence: {pred_intact['global_confidence']:.3f}")
        print(f"  Spike activity: {pred_intact['spike_activity']}")
        
        print(f"\nDamaged sample prediction:")
        print(f"  Damage: {'Damaged' if pred_damaged['global_damage'] == 1 else 'Intact'}")
        print(f"  Confidence: {pred_damaged['global_confidence']:.3f}")
        print(f"  Spike activity: {pred_damaged['spike_activity']}")
        
        # Visualize results
        print(f"\nFeature comparison:")
        print(f"Intact features sum: {np.sum(intact_features)}")
        print(f"Damaged features sum: {np.sum(damaged_features)}")

def main():
    """Run all demos."""
    print("🚀 SPIKING GRAPH NEURAL NETWORKS DEMO")
    print("🔬 Testing individual components and integrated functionality")
    print("📊 Using synthetic data for demonstration")
    
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Run individual component demos
        intact_features, damaged_features = demo_binary_feature_extraction()
        graph_builder = demo_graph_topology()
        model = demo_sgnn_model()
        
        # Run integrated demo
        demo_integrated_pipeline()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("🎯 Next steps:")
        print("   1. Run 'python sgnn_pipeline.py' for full training pipeline")
        print("   2. Place your Excel data files in the current directory")
        print("   3. Adjust hyperparameters in the pipeline for your data")
        print("   4. Explore different graph topologies and model architectures")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check the installation and requirements.")

if __name__ == "__main__":
    main() 