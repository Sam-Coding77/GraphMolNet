# GraphMolNet Project - README

# Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
   - [Additional Dependencies](#additional-dependencies)
3. [Dataset](#dataset)
4. [Graph Visualization](#graph-visualization)
5. [Preprocessing](#preprocessing)
6. [GCN Model](#gcn-model)
7. [Training and Testing](#training-and-testing)
8. [Results and Visualization](#results-and-visualization)
   - [Prediction Visualization](#prediction-visualization)
   - [Test Graph Structure Visualization](#test-graph-structure-visualization)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)
11. [References](#references)

## Introduction

This project demonstrates the use of Graph Neural Networks (GNNs) for molecular/protein structure classification using the `graphs-datasets/PROTEINS` dataset. It utilizes `PyTorch Geometric` to construct a Graph Convolutional Network (GCN) model for learning graph representations of proteins and predicting their classification.

## Installation

To run this project, you will need to install the required dependencies:

```bash
pip install transformers
pip install datasets
pip install rdkit-pypi
pip install torch-geometric
```

### Additional Dependencies

- `torch`: For running PyTorch deep learning models.
- `networkx`: For visualizing graphs.
- `matplotlib`: For plotting and visualizing the results.
- `rdkit`: For handling molecule-specific computations.

## Dataset

The dataset used is `graphs-datasets/PROTEINS`, a common graph dataset for classification tasks. It consists of graph-structured data, where each node represents part of a molecule or protein, and the edges represent the connections between them.

The dataset is loaded using the `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("graphs-datasets/PROTEINS")
```

## Graph Visualization

The project includes a function to visualize a graph using `NetworkX`:

```python
def visualize_graph(edge_index, num_nodes):
    # Visualizes the structure of the graph
```

## Preprocessing

Each graph in the dataset consists of `edge_index` (defining the structure) and `node_feat` (features of the nodes). These are converted into PyTorch tensors:

```python
def preprocess_graph_data(sample):
    # Converts edge_index, node_feat, and labels into tensors
```

The dataset is split into training and test sets:

```python
train_test_split = ds['train'].train_test_split(test_size=0.2)
```

## GCN Model

The project uses a Graph Convolutional Network (GCN) to process the graph data. The GCN model is implemented with three graph convolutional layers followed by global mean pooling, which pools the graph's node features:

```python
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        # Initializes the GCN with convolution layers and linear output layer
```

The model is trained using Negative Log Likelihood (NLL) loss and Adam optimizer.

## Training and Testing

Training and testing loops are implemented as follows:

```python
def train(model, optimizer, data_loader):
    # Trains the model on the provided data loader

def test(model, data_loader):
    # Evaluates the model's accuracy
```

The training runs for 100 epochs, printing the loss and accuracy at each step.

```python
for epoch in range(1, num_epochs+1):
    # Trains the model and prints accuracy for each epoch
```

## Results and Visualization

To visualize the performance of the model, predictions are plotted against true labels:

```python
def visualize_predictions(model, data_loader, num_samples=10):
    # Visualizes predictions vs true labels
```

Additionally, the structure of test graphs is visualized with predicted and true labels annotated:

```python
def visualize_test_graph_structures(model, data_loader, num_graphs=5):
    # Visualizes a few test graphs with predictions
```

## Conclusion

This project demonstrates how to build and train a GCN for molecular or protein structure classification. The visualization tools provided help to understand the structure of the data and evaluate the performance of the model.

## Future Work

- Explore different GNN architectures (e.g., GraphSAGE, GAT).
- Implement hyperparameter tuning to improve model accuracy.
- Experiment with larger datasets or more complex molecular structures.
  
## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io)
- [NetworkX Documentation](https://networkx.github.io/)
- [RDKit Documentation](https://www.rdkit.org/)

