# Source Code

This folder contains the core source code for the multimodal benchmark framework, including model definitions, training pipelines, and evaluation scripts.

## Contents
- `main.py`: Choosing downstream tasks and hyperparameters
- `/lp`: Link prediction source code
- `/nc`: Node classification source code
- `/gc`: Graph classification source code

## Models
### Model List

The following graph neural network models are supported by the framework:

- GCN
- GraphSAGE
- GAT
- MLP
- MMGCN
- MGAT
- RevGAT

### Importing and Using Models

You can import and use these models as follows:

```python
from model.models import GCN, GraphSAGE, GAT, MLP
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
```

### Model Construction Example

Here is an example of constructing a model using **GCN**:

```python
if config.model.name == "GCN":
    encoder = GCN(
        in_dim=data.x.size(1),     # Input feature dimension
        hidden_dim=64,              # Hidden layer dimension
        num_layers=3,              # Number of layers
        dropout=0.5                # Dropout rate
    )
```

#### Parameter Explanation:

- `in_dim`: The dimension of the input node features (`data.x.size(1)`).
- `hidden_dim`: The dimension of the hidden layers, set to 64.
- `num_layers`: The number of layers in the model, set to 3.
- `dropout`: The dropout rate, set to 0.5.

### Model's `reset_parameters` Method

All models provide the `reset_parameters` method to reset the model's parameters, which is useful when training multiple times. You can call `reset_parameters()` to reinitialize the model's weights.

```python
# Reset model parameters
encoder.reset_parameters()
```

### Model Calling Example

During training and inference, you can call the model for forward propagation as follows:

```python
# Model forward pass
out, out_v, out_t = encoder(data.x, data.edge_index)
```

#### Input Explanation:

- `data.x`: The input node feature matrix, with shape `[num_nodes, feature_dim]`.
- `data.edge_index`: The edge index representing the graph's topology, with shape `[2, num_edges]`.

#### Output Explanation:

- `out`: Node embeddings, with shape `[num_nodes, hidden_dim]`.
- `out_v`: Embeddings for the first modality, with shape `[num_nodes, hidden_dim]`.
- `out_t`: Embeddings for the second modality, with shape `[num_nodes, hidden_dim]`.

These three outputs represent the node embeddings and the embeddings for the two modalities in the graph.

