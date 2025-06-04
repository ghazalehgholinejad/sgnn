import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from binary_feature_extractor import BinaryFeatureExtractor
from graph_topology import SensorGraphTopology
from spiking_gnn_model import SpikingGNN, SGNNTrainer

class SGNNDataset:
    """
    Dataset class for SGNN damage detection.
    Handles data loading, feature extraction, and graph construction.
    """
    
    def __init__(self, 
                 extractor: BinaryFeatureExtractor,
                 graph_builder: SensorGraphTopology):
        self.extractor = extractor
        self.graph_builder = graph_builder
        self.graphs = []
        self.labels = []
        
    def add_sample(self, strain_data: np.ndarray, label: int, 
                   node_labels: Optional[np.ndarray] = None):
        """
        Add a single sample to the dataset.
        
        Args:
            strain_data: Raw strain data (timesteps, sensors)
            label: Global damage label (0: intact, 1: damaged)
            node_labels: Optional node-level damage labels (sensors,)
        """
        # Extract binary features
        binary_features = self.extractor.extract_features_all_sensors(strain_data)
        
        # Create PyTorch Geometric Data object
        x = torch.tensor(binary_features, dtype=torch.float32)
        edge_index = self.graph_builder.edge_index
        edge_weight = self.graph_builder.edge_weights
        y = torch.tensor([label], dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        
        if node_labels is not None:
            graph_data.node_labels = torch.tensor(node_labels, dtype=torch.float32)
        
        self.graphs.append(graph_data)
        self.labels.append(label)
    
    def create_augmented_dataset(self, 
                               intact_strain: np.ndarray,
                               damaged_strain: np.ndarray,
                               num_augmentations: int = 50):
        """
        Create an augmented dataset with noise variations.
        
        Args:
            intact_strain: Intact sensor data (timesteps, sensors)
            damaged_strain: Damaged sensor data (timesteps, sensors)
            num_augmentations: Number of augmented samples per original sample
        """
        print(f"Creating augmented dataset with {num_augmentations} variations per sample...")
        
        # Node labels for intact samples (all zeros)
        intact_node_labels = np.zeros(9)
        
        # Original intact sample
        self.add_sample(intact_strain, label=0, node_labels=intact_node_labels)
        
        # Original damaged sample
        damage_center = 4  # Assume damage near center sensor (sensor 5, index 4)
        damaged_node_labels = np.zeros(9)
        damaged_node_labels[damage_center] = 1  # Mark center as damaged
        # Also mark nearby sensors as potentially affected
        for neighbor in [1, 3, 5, 7]:  # Adjacent to center in 3x3 grid
            damaged_node_labels[neighbor] = 0.5  # Partially affected
        
        self.add_sample(damaged_strain, label=1, node_labels=damaged_node_labels)
        
        # Generate augmentations
        for i in range(num_augmentations):
            # Intact augmentations - always provide node_labels (all zeros)
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, intact_strain.shape)
            augmented_intact = intact_strain + noise
            self.add_sample(augmented_intact, label=0, node_labels=intact_node_labels)
            
            # Damaged augmentations
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, damaged_strain.shape)
            augmented_damaged = damaged_strain + noise
            self.add_sample(augmented_damaged, label=1, node_labels=damaged_node_labels)
            
            # Additional damaged variations with different damage locations
            if i < num_augmentations // 2:
                # Simulate damage at different locations
                damage_location = np.random.choice([0, 2, 6, 8])  # Corner sensors
                varied_node_labels = np.zeros(9)
                varied_node_labels[damage_location] = 1
                
                # Add some random strain amplification at damage location
                damage_amplification = np.random.uniform(1.2, 2.0)
                augmented_damaged_varied = damaged_strain.copy()
                augmented_damaged_varied[:, damage_location] *= damage_amplification
                augmented_damaged_varied += noise
                
                self.add_sample(augmented_damaged_varied, label=1, node_labels=varied_node_labels)
        
        print(f"Dataset created with {len(self.graphs)} samples")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        # Verify all samples have node_labels
        samples_with_node_labels = sum(1 for graph in self.graphs if hasattr(graph, 'node_labels'))
        print(f"Samples with node_labels: {samples_with_node_labels}/{len(self.graphs)}")
    
    def get_data_loaders(self, 
                        batch_size: int = 16,
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Split indices
        train_val_idx, test_idx = train_test_split(
            range(len(self.graphs)), 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        train_labels = [self.labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=train_labels
        )
        
        # Create datasets
        train_data = [self.graphs[i] for i in train_idx]
        val_data = [self.graphs[i] for i in val_idx]
        test_data = [self.graphs[i] for i in test_idx]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_loader, val_loader, test_loader


class SGNNPipeline:
    """
    Complete pipeline for SGNN-based damage detection.
    """
    
    def __init__(self, 
                 num_sensors: int = 9,
                 graph_type: str = "delaunay",
                 model_params: Optional[Dict] = None):
        """
        Initialize the SGNN pipeline.
        
        Args:
            num_sensors: Number of sensors in the network
            graph_type: Type of graph topology ("grid", "delaunay", "knn", "radius")
            model_params: Dictionary of model parameters
        """
        self.num_sensors = num_sensors
        self.graph_type = graph_type
        
        # Initialize components
        self.feature_extractor = BinaryFeatureExtractor()
        self.graph_builder = SensorGraphTopology(num_sensors=num_sensors)
        
        # Set up sensor positions and graph topology
        self._setup_graph_topology()
        
        # Model parameters
        default_params = {
            'num_features': 5,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_classes': 2,
            'lif_threshold': 1.0,
            'lif_decay': 0.9,
            'dropout': 0.1
        }
        self.model_params = {**default_params, **(model_params or {})}
        
        # Initialize model and trainer
        self.model = None
        self.trainer = None
        self.dataset = None
        
    def _setup_graph_topology(self):
        """Set up graph topology based on the specified type."""
        # Set sensor positions (3x3 grid)
        positions = self.graph_builder.set_sensor_positions_3x3_grid(spacing=1.0)
        
        # Create graph topology
        if self.graph_type == "grid":
            edge_index, edge_weights = self.graph_builder.create_grid_adjacency()
        elif self.graph_type == "delaunay":
            edge_index, edge_weights = self.graph_builder.create_delaunay_triangulation()
        elif self.graph_type == "knn":
            edge_index, edge_weights = self.graph_builder.create_knn_graph(k=3)
        elif self.graph_type == "radius":
            edge_index, edge_weights = self.graph_builder.create_radius_graph(radius=1.5)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        print(f"Graph topology ({self.graph_type}) created:")
        self.graph_builder.print_graph_info()
    
    def load_and_prepare_data(self, 
                            intact_file: str,
                            damaged_file: str,
                            num_augmentations: int = 100):
        """
        Load data from Excel files and prepare the dataset.
        
        Args:
            intact_file: Path to intact sensor data Excel file
            damaged_file: Path to damaged sensor data Excel file
            num_augmentations: Number of data augmentations to create
        """
        print("Loading and preparing data...")
        
        # Load raw strain data
        intact_strain, intact_features = self.feature_extractor.load_and_process_excel(intact_file)
        damaged_strain, damaged_features = self.feature_extractor.load_and_process_excel(damaged_file)
        
        if intact_strain is None or damaged_strain is None:
            raise ValueError("Failed to load strain data from Excel files")
        
        # Compare features
        stats = self.feature_extractor.compare_intact_vs_damaged(intact_features, damaged_features)
        self.feature_extractor.print_comparison_report(stats)
        
        # Visualize binary features
        self.feature_extractor.visualize_features(intact_features, "Intact State - Binary Features")
        self.feature_extractor.visualize_features(damaged_features, "Damaged State - Binary Features")
        
        # Create dataset
        self.dataset = SGNNDataset(self.feature_extractor, self.graph_builder)
        self.dataset.create_augmented_dataset(intact_strain, damaged_strain, num_augmentations)
        
        return self.dataset
    
    def create_model(self):
        """Create and initialize the SGNN model."""
        self.model = SpikingGNN(
            num_sensors=self.num_sensors,
            **self.model_params
        )
        
        print(f"SGNN Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Model architecture: {self.model}")
        
        return self.model
    
    def train_model(self, 
                   num_epochs: int = 100,
                   learning_rate: float = 0.001,
                   batch_size: int = 16,
                   device: Optional[torch.device] = None):
        """
        Train the SGNN model.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device to train on (CPU/GPU)
        """
        if self.model is None:
            self.create_model()
        
        if self.dataset is None:
            raise ValueError("Dataset not prepared. Call load_and_prepare_data() first.")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.dataset.get_data_loaders(batch_size=batch_size)
        
        # Set up device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize trainer
        self.trainer = SGNNTrainer(self.model, device)
        
        # Set up optimizer and loss functions
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion_global = nn.CrossEntropyLoss()
        criterion_node = nn.BCELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        print(f"Training on device: {device}")
        print(f"Training for {num_epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        patience = 20
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.trainer.train_epoch(
                train_loader, optimizer, criterion_global, criterion_node
            )
            
            # Validate
            val_loss, val_acc = self.trainer.validate(
                val_loader, criterion_global, criterion_node
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_sgnn_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_sgnn_model.pth'))
        
        # Final evaluation on test set
        test_loss, test_acc = self.trainer.validate(test_loader, criterion_global, criterion_node)
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        
        # Plot training history
        self.trainer.plot_training_history()
        
        return self.model
    
    def evaluate_model(self, test_loader: DataLoader):
        """
        Comprehensive evaluation of the trained model.
        
        Args:
            test_loader: Test data loader
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_node_probs = []
        all_spike_counts = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.trainer.device)
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Global predictions
                preds = torch.argmax(outputs['global_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                
                # Node predictions and spike activity
                all_node_probs.extend(outputs['node_probs'].cpu().numpy())
                all_spike_counts.extend([outputs['spike_counts']])
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Intact', 'Damaged']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Intact', 'Damaged'],
                   yticklabels=['Intact', 'Damaged'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Spike activity analysis
        avg_spike_counts = np.mean(all_spike_counts, axis=0)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_spike_counts)), avg_spike_counts)
        plt.title('Average Spike Activity per Layer')
        plt.xlabel('Layer')
        plt.ylabel('Average Spike Count')
        plt.show()
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'node_probabilities': all_node_probs,
            'spike_counts': all_spike_counts
        }
    
    def visualize_damage_detection(self, sample_data: Data):
        """
        Visualize damage detection results on a sample.
        
        Args:
            sample_data: A single graph sample
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        predictions = self.model.predict(sample_data.x, sample_data.edge_index, sample_data.edge_attr)
        
        print(f"Global Damage Prediction: {'Damaged' if predictions['global_damage'] == 1 else 'Intact'}")
        print(f"Confidence: {predictions['global_confidence']:.3f}")
        print(f"Spike Activity: {predictions['spike_activity']}")
        
        # Visualize graph with node damage probabilities
        self.graph_builder.visualize_graph(
            title=f"Damage Detection Results (Global: {'Damaged' if predictions['global_damage'] == 1 else 'Intact'})",
            node_features=sample_data.x.numpy()
        )
        
        # Plot node damage probabilities
        plt.figure(figsize=(10, 6))
        sensor_ids = [f'S{i+1}' for i in range(self.num_sensors)]
        plt.bar(sensor_ids, predictions['node_damage_probs'])
        plt.title('Node-level Damage Probabilities')
        plt.xlabel('Sensor')
        plt.ylabel('Damage Probability')
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example usage of the complete pipeline
    print("="*60)
    print("SPIKING GRAPH NEURAL NETWORK DAMAGE DETECTION PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = SGNNPipeline(
        num_sensors=9,
        graph_type="delaunay",
        model_params={
            'hidden_dim': 64,
            'num_layers': 3,
            'lif_threshold': 1.0,
            'lif_decay': 0.9
        }
    )
    
    # Load and prepare data
    try:
        dataset = pipeline.load_and_prepare_data(
            intact_file='AllSensors-Intact.xlsx',
            damaged_file='AllSensors-Damaged.xlsx',
            num_augmentations=100
        )
        
        # Train model
        model = pipeline.train_model(
            num_epochs=100,
            learning_rate=0.001,
            batch_size=16
        )
        
        # Evaluate model
        train_loader, val_loader, test_loader = dataset.get_data_loaders()
        results = pipeline.evaluate_model(test_loader)
        
        # Visualize a sample prediction
        sample_graph = dataset.graphs[1]  # Damaged sample
        pipeline.visualize_damage_detection(sample_graph)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        print("Make sure the Excel files are available in the current directory.") 