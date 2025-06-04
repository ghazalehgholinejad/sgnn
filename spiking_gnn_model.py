import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron implementation for spiking neural networks.
    
    Dynamics:
    - V(t+1) = decay * V(t) + I(t) - reset * spike(t)
    - spike(t) = 1 if V(t) >= threshold else 0
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 reset_voltage: float = 0.0,
                 refractory_period: int = 1):
        super(LIFNeuron, self).__init__()
        
        self.threshold = threshold
        self.decay = decay
        self.reset_voltage = reset_voltage
        self.refractory_period = refractory_period
        
        # State variables (will be set during forward pass)
        self.voltage = None
        self.refractory_counter = None
        
    def reset_state(self, batch_size: int, num_neurons: int, device: torch.device):
        """Reset neuron states for new sequence."""
        self.voltage = torch.zeros(batch_size, num_neurons, device=device)
        self.refractory_counter = torch.zeros(batch_size, num_neurons, device=device, dtype=torch.int)
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LIF neuron.
        
        Args:
            input_current: Input current (batch_size, num_neurons)
            
        Returns:
            spikes: Binary spike output (batch_size, num_neurons)
        """
        if self.voltage is None:
            batch_size, num_neurons = input_current.shape
            self.reset_state(batch_size, num_neurons, input_current.device)
        
        # Update voltage (only if not in refractory period)
        not_refractory = (self.refractory_counter == 0)
        voltage_update = self.decay * self.voltage + input_current
        self.voltage = torch.where(not_refractory, voltage_update, self.voltage)
        
        # Generate spikes
        spikes = (self.voltage >= self.threshold).float()
        
        # Reset voltage for spiking neurons
        self.voltage = torch.where(spikes.bool(), 
                                 torch.full_like(self.voltage, self.reset_voltage),
                                 self.voltage)
        
        # Update refractory counters
        self.refractory_counter = torch.where(spikes.bool(),
                                            torch.full_like(self.refractory_counter, self.refractory_period),
                                            torch.maximum(self.refractory_counter - 1, 
                                                        torch.zeros_like(self.refractory_counter)))
        
        return spikes


class SpikingGraphConv(MessagePassing):
    """
    Spiking Graph Convolutional Layer that combines message passing with LIF neurons.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 lif_threshold: float = 1.0,
                 lif_decay: float = 0.9,
                 aggr: str = 'add'):
        super(SpikingGraphConv, self).__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation for messages
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        
        # LIF neurons
        self.lif = LIFNeuron(threshold=lif_threshold, decay=lif_decay)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of spiking graph convolution.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            
        Returns:
            spikes: Spiking output (num_nodes, out_channels)
        """
        # Message passing
        messages = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Self-connection
        self_features = self.lin_self(x)
        
        # Combine messages and self-features as input current
        input_current = messages + self_features
        
        # Reset LIF neuron states based on actual input size
        num_nodes_total, num_features = input_current.shape
        device = input_current.device
        
        # Reset LIF neurons for the current batch
        self.lif.reset_state(1, num_nodes_total * self.out_channels, device)
        
        # Pass through LIF neurons
        # Add batch dimension for LIF neuron (batch_size=1, nodes*features)
        input_current_flat = input_current.view(1, -1)
        spikes_flat = self.lif(input_current_flat)
        
        # Reshape back to (num_nodes, out_channels)
        spikes = spikes_flat.view(num_nodes_total, self.out_channels)
        
        return spikes
    
    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute messages from neighboring nodes."""
        messages = self.lin(x_j)
        if edge_weight is not None:
            messages = messages * edge_weight.view(-1, 1)
        return messages
    
    def reset_neuron_state(self, batch_size: int, num_nodes: int, device: torch.device):
        """Reset LIF neuron states - now handled in forward pass."""
        # This method is kept for compatibility but actual reset is done in forward()
        pass


class SpikingGNN(nn.Module):
    """
    Spiking Graph Neural Network for damage detection in sensor networks.
    
    Architecture:
    - Input encoding layer
    - Multiple spiking graph convolutional layers
    - Global pooling and classification layers
    """
    
    def __init__(self,
                 num_sensors: int = 9,
                 num_features: int = 5,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 lif_threshold: float = 1.0,
                 lif_decay: float = 0.9,
                 dropout: float = 0.1):
        super(SpikingGNN, self).__init__()
        
        self.num_sensors = num_sensors
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input encoding layer
        self.input_encoder = nn.Linear(num_features, hidden_dim)
        
        # Spiking graph convolutional layers
        self.spiking_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.spiking_layers.append(
                SpikingGraphConv(in_dim, hidden_dim, lif_threshold, lif_decay)
            )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # For node-level predictions (damage localization)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Spiking GNN.
        
        Args:
            x: Node features (num_nodes, num_features)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            batch: Batch assignment for each node (num_nodes,)
            
        Returns:
            Dictionary containing:
            - global_logits: Global damage classification (batch_size, num_classes)
            - node_probs: Node-level damage probabilities (num_nodes, 1)
            - spike_counts: Spike counts per layer for analysis
        """
        batch_size = 1 if batch is None else batch.max().item() + 1
        device = x.device
        
        # Input encoding
        h = self.input_encoder(x)
        
        # Store spike counts for analysis
        spike_counts = []
        
        # Forward through spiking layers (each layer handles its own neuron state reset)
        for layer in self.spiking_layers:
            h = layer(h, edge_index, edge_weight)
            spike_counts.append(h.sum().item())
        
        # Node-level predictions (damage localization)
        node_probs = self.node_classifier(h)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            # Batch processing
            from torch_geometric.nn import global_mean_pool
            global_features = global_mean_pool(h, batch)
        else:
            # Single graph
            global_features = h.mean(dim=0, keepdim=True)
        
        global_features = self.dropout(global_features)
        global_logits = self.classifier(global_features)
        
        return {
            'global_logits': global_logits,
            'node_probs': node_probs,
            'spike_counts': spike_counts,
            'node_features': h
        }
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Make predictions with the trained model.
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, edge_index, edge_weight)
            
            # Global prediction
            global_probs = F.softmax(outputs['global_logits'], dim=1)
            global_pred = torch.argmax(global_probs, dim=1)
            
            # Node predictions
            node_preds = (outputs['node_probs'] > 0.5).float()
            
            return {
                'global_damage': global_pred.item(),
                'global_confidence': global_probs.max().item(),
                'node_damage_probs': outputs['node_probs'].squeeze().cpu().numpy(),
                'node_damage_binary': node_preds.squeeze().cpu().numpy(),
                'spike_activity': outputs['spike_counts']
            }


class SGNNTrainer:
    """
    Training utilities for Spiking Graph Neural Networks.
    """
    
    def __init__(self, model: SpikingGNN, device: torch.device = None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, optimizer, criterion_global, criterion_node):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Global classification loss
            global_loss = criterion_global(outputs['global_logits'], batch.y)
            
            # Node classification loss (if available and properly shaped)
            node_loss = 0
            if hasattr(batch, 'node_labels') and batch.node_labels is not None:
                try:
                    # Ensure node_labels has the right shape for loss computation
                    node_probs = outputs['node_probs'].squeeze()
                    node_labels = batch.node_labels.float()
                    
                    # Check shapes match
                    if node_probs.shape == node_labels.shape:
                        node_loss = criterion_node(node_probs, node_labels)
                    else:
                        print(f"Warning: Shape mismatch in node loss - probs: {node_probs.shape}, labels: {node_labels.shape}")
                except Exception as e:
                    print(f"Warning: Could not compute node loss: {e}")
            
            total_loss_batch = global_loss + 0.5 * node_loss  # Weight node loss
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Accuracy calculation
            pred = torch.argmax(outputs['global_logits'], dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion_global, criterion_node):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Global classification loss
                global_loss = criterion_global(outputs['global_logits'], batch.y)
                
                # Node classification loss (if available and properly shaped)
                node_loss = 0
                if hasattr(batch, 'node_labels') and batch.node_labels is not None:
                    try:
                        # Ensure node_labels has the right shape for loss computation
                        node_probs = outputs['node_probs'].squeeze()
                        node_labels = batch.node_labels.float()
                        
                        # Check shapes match
                        if node_probs.shape == node_labels.shape:
                            node_loss = criterion_node(node_probs, node_labels)
                        else:
                            print(f"Warning: Shape mismatch in node loss - probs: {node_probs.shape}, labels: {node_labels.shape}")
                    except Exception as e:
                        print(f"Warning: Could not compute node loss: {e}")
                
                total_loss_batch = global_loss + 0.5 * node_loss
                total_loss += total_loss_batch.item()
                
                # Accuracy calculation
                pred = torch.argmax(outputs['global_logits'], dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def plot_training_history(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Creating Spiking GNN for damage detection...")
    
    # Model parameters
    num_sensors = 9
    num_features = 5
    hidden_dim = 32
    num_layers = 2
    
    # Create model
    model = SpikingGNN(
        num_sensors=num_sensors,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lif_threshold=1.0,
        lif_decay=0.9
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data for testing
    x = torch.randn(num_sensors, num_features)  # Node features
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], 
                              [1, 0, 2, 1, 4, 3]], dtype=torch.long)  # Example edges
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, edge_index)
        print(f"Global logits shape: {outputs['global_logits'].shape}")
        print(f"Node probabilities shape: {outputs['node_probs'].shape}")
        print(f"Spike counts per layer: {outputs['spike_counts']}")
    
    print("SGNN model created successfully!") 