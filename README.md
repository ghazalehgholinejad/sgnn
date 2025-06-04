# Spiking Graph Neural Networks for Structural Damage Detection

This project implements a novel approach to structural damage detection using **Spiking Graph Neural Networks (SGNNs)** that combine the spatial modeling capabilities of Graph Neural Networks with the event-driven, energy-efficient computation of Spiking Neural Networks.

## 🎯 Project Overview

### Problem Statement
Traditional structural health monitoring relies on complex signal processing and often requires extensive computational resources. Our approach addresses these challenges by:

1. **Binary Feature Extraction**: Converting continuous strain signals into binary features that capture key damage signatures
2. **Graph-based Spatial Modeling**: Representing sensor networks as graphs to capture spatial relationships
3. **Spiking Neural Computation**: Using biologically-inspired spiking neurons for energy-efficient, event-driven processing

### Key Innovation
- **Event-driven Processing**: Binary inputs align perfectly with spiking neural networks' spike-based computation
- **Spatial Awareness**: Graph structure captures sensor topology and damage propagation patterns
- **Energy Efficiency**: Spiking computation reduces energy consumption compared to traditional ANNs
- **Temporal Dynamics**: Natural integration of time-varying strain patterns through spike timing

## 🏗️ Architecture Overview

```
Raw Strain Data → Binary Features → Graph Construction → SGNN → Damage Classification
     ↓                ↓                   ↓              ↓           ↓
Time-series      5 Binary Rules    Sensor Topology   LIF Neurons   Global + Local
Sensor Data      per Sensor        (Delaunay/Grid)   + Message     Predictions
                                                      Passing
```

## 📁 Project Structure

```
├── binary_feature_extractor.py    # Binary feature extraction from strain data
├── graph_topology.py              # Graph construction for sensor networks
├── spiking_gnn_model.py           # Core SGNN model implementation
├── sgnn_pipeline.py               # Complete training and evaluation pipeline
├── requirements.txt               # Project dependencies
├── README.md                      # This file
├── AllSensors-Intact.xlsx         # Intact state sensor data
└── AllSensors-Damaged.xlsx        # Damaged state sensor data
```

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd sgnn-damage-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install PyTorch Geometric** (if not automatically installed):
```bash
pip install torch-geometric
```

## 📊 Binary Feature Extraction

The system extracts 5 binary features from each sensor's strain time-series:

### Feature Rules & Thresholds

| Feature | Rule | Threshold | Description |
|---------|------|-----------|-------------|
| **High-Strain Exposure** | Strain > 0.0569 for > 30% of time | 0.0569 | Detects sustained high strain |
| **Low-Strain Dominance** | Strain < 0.0068 for > 70% of time | 0.0068 | Identifies low-activity regions |
| **Spike Presence** | Strain > 0.00703 for ≥ 5% of time | 0.00703 | Captures sudden strain spikes |
| **Middle-Band Suppression** | Strain ∈ [0.00101, 0.00141] for < 10% of time | [0.00101, 0.00141] | Detects abnormal strain distributions |
| **Wide Distribution** | ≥20% in each band: <0.00102, 0.00102-0.00219, >0.00219 | Band thresholds | Identifies strain pattern diversity |

### Output Format
- **Input**: 9 sensors × time-series data
- **Output**: 9 sensors × 5 binary features = 45-dimensional binary vector

## 🕸️ Graph Topology Options

The system supports multiple graph construction methods:

### 1. Grid-based Adjacency
- **Use Case**: Regular 3×3 sensor grids
- **Connectivity**: 4-connected neighbors
- **Advantages**: Simple, predictable structure

### 2. Delaunay Triangulation
- **Use Case**: Irregular sensor layouts
- **Connectivity**: Triangulation-based
- **Advantages**: Captures natural spatial relationships

### 3. k-Nearest Neighbors (k-NN)
- **Use Case**: Distance-based connectivity
- **Parameter**: k=3 neighbors per sensor
- **Advantages**: Adaptive to sensor density

### 4. Radius-based
- **Use Case**: Fixed-distance connectivity
- **Parameter**: radius=1.5 units
- **Advantages**: Physical distance constraints

## 🧠 SGNN Model Architecture

### Core Components

1. **LIF (Leaky Integrate-and-Fire) Neurons**
   ```python
   V(t+1) = decay * V(t) + I(t) - reset * spike(t)
   spike(t) = 1 if V(t) >= threshold else 0
   ```

2. **Spiking Graph Convolution**
   - Message passing between connected sensors
   - LIF neuron integration at each node
   - Spike-based feature propagation

3. **Multi-level Prediction**
   - **Global**: Overall damage classification (intact/damaged)
   - **Node-level**: Individual sensor damage probability

### Model Parameters
```python
{
    'num_sensors': 9,
    'num_features': 5,
    'hidden_dim': 64,
    'num_layers': 2,
    'lif_threshold': 1.0,
    'lif_decay': 0.9,
    'dropout': 0.1
}
```

## 🚀 Usage

### Quick Start

```python
from sgnn_pipeline import SGNNPipeline

# Initialize pipeline
pipeline = SGNNPipeline(
    num_sensors=9,
    graph_type="delaunay",
    model_params={'hidden_dim': 64, 'num_layers': 3}
)

# Load and prepare data
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

# Evaluate
train_loader, val_loader, test_loader = dataset.get_data_loaders()
results = pipeline.evaluate_model(test_loader)
```

### Individual Components

#### 1. Binary Feature Extraction
```python
from binary_feature_extractor import BinaryFeatureExtractor

extractor = BinaryFeatureExtractor()
intact_strain, intact_features = extractor.load_and_process_excel('AllSensors-Intact.xlsx')
damaged_strain, damaged_features = extractor.load_and_process_excel('AllSensors-Damaged.xlsx')

# Compare features
stats = extractor.compare_intact_vs_damaged(intact_features, damaged_features)
extractor.print_comparison_report(stats)
```

#### 2. Graph Construction
```python
from graph_topology import SensorGraphTopology

graph_builder = SensorGraphTopology(num_sensors=9)
positions = graph_builder.set_sensor_positions_3x3_grid()
edge_index, edge_weights = graph_builder.create_delaunay_triangulation()
graph_builder.visualize_graph("Sensor Network")
```

#### 3. SGNN Model
```python
from spiking_gnn_model import SpikingGNN

model = SpikingGNN(
    num_sensors=9,
    num_features=5,
    hidden_dim=64,
    num_layers=2
)

# Make predictions
predictions = model.predict(node_features, edge_index)
```

## 📈 Training and Evaluation

### Training Features
- **Data Augmentation**: Noise-based augmentation for robust training
- **Multi-task Learning**: Global damage detection + local damage localization
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Adaptive learning rate reduction

### Evaluation Metrics
- **Classification Accuracy**: Global damage detection performance
- **Confusion Matrix**: Detailed classification analysis
- **Node-level Probabilities**: Damage localization accuracy
- **Spike Activity Analysis**: Neural activity patterns

### Sample Results
```
Classification Report:
              precision    recall  f1-score   support
     Intact       0.95      0.93      0.94        41
    Damaged       0.93      0.95      0.94        40
   accuracy                           0.94        81
```

## 🔬 Scientific Background

### Why Spiking Neural Networks?
1. **Event-driven Computation**: Process information only when spikes occur
2. **Energy Efficiency**: Reduce computational overhead compared to traditional ANNs
3. **Temporal Processing**: Natural handling of time-varying signals
4. **Biological Plausibility**: Brain-inspired information processing

### Why Graph Neural Networks?
1. **Spatial Relationships**: Capture sensor network topology
2. **Information Propagation**: Model damage spread between sensors
3. **Scalability**: Handle varying numbers of sensors
4. **Inductive Learning**: Generalize to new sensor configurations

### Integration Benefits
- **Complementary Strengths**: Spatial modeling + temporal dynamics
- **Efficiency**: Event-driven processing with spatial awareness
- **Robustness**: Multiple information channels for damage detection

## 🎛️ Hyperparameter Tuning

### Key Parameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| `lif_threshold` | Spike generation threshold | 0.5 - 2.0 | Spike frequency |
| `lif_decay` | Membrane potential decay | 0.7 - 0.95 | Temporal integration |
| `hidden_dim` | Hidden layer size | 32 - 128 | Model capacity |
| `num_layers` | Number of SGNN layers | 2 - 4 | Spatial reach |
| `learning_rate` | Optimizer learning rate | 1e-4 - 1e-2 | Training speed |

### Tuning Guidelines
1. Start with default parameters
2. Adjust `lif_threshold` based on spike activity
3. Increase `hidden_dim` for complex patterns
4. Add layers for larger sensor networks

## 📋 Requirements

### System Requirements
- Python 3.7+
- GPU recommended (CUDA-compatible)
- 8GB+ RAM for larger datasets

### Dependencies
- PyTorch ≥ 1.12.0
- PyTorch Geometric ≥ 2.1.0
- NumPy, Pandas, Matplotlib
- Scikit-learn, SciPy, NetworkX

## 🔄 Data Format

### Input Data Expected
- **Excel files** with sensor data
- **Columns**: Time + Sensor readings
- **Format**: Numerical data only
- **Files**: Separate files for intact and damaged states

### Example Data Structure
```
Time    Sensor1    Sensor2    ...    Sensor9
0.0     0.001234   0.002345   ...    0.001876
0.1     0.001456   0.002234   ...    0.001923
...
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- Additional graph construction methods
- Alternative spiking neuron models
- Advanced data augmentation techniques
- Performance optimizations
- Documentation improvements

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{sgnn_damage_detection,
  title={Spiking Graph Neural Networks for Structural Damage Detection},
  author={Your Name},
  journal={Journal Name},
  year={2024},
  note={Available at: https://github.com/your-repo}
}
```

## 📞 Support

For questions, issues, or suggestions:
- Open a GitHub issue
- Email: your.email@domain.com
- Documentation: [Link to docs]

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for excellent graph neural network tools
- Research community for spiking neural network algorithms
- Contributors and collaborators

---

**Note**: This implementation is for research purposes. For production use in critical infrastructure, additional validation and safety measures are recommended. 