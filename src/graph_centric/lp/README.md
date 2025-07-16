Here's a markdown template for the **Link Prediction (LP)** task, following the same structure as your **Node Classification (NC)** README but with the appropriate changes for the Link Prediction task configuration.

~~~markdown
# Link Prediction (LP)

This folder contains the source code for link prediction (LP) tasks using graph neural networks (GNNs). It supports both full-graph training and batch-based subgraph training.

## Task Overview

The link prediction task involves predicting the existence of edges between pairs of nodes in a graph. The framework supports different GNN models like GCN and different datasets for testing.

## Available Files

- `run.py`: This script runs the model on the entire graph.
- `run_batch.py`: This script runs the model on subgraphs in batches.

## Configuration Format

The configuration is defined in YAML format, where we use Hydra for configuration management. The main configuration file `config.yaml` includes multiple sections, such as the task, model, dataset, and other training settings.

### Example Config File (`config.yaml`)

```yaml
defaults:
  - task: lp
  - model: gcn
  - dataset: default
  - _self_

# Global Settings
seed: 42
log_dir: outputs/
device: cuda

# Link Prediction Task Settings
task:
  name: lp
  n_epochs: 1000
  lr: 0.001
  fewshots: False
  weight_decay: 1e-5
  n_runs: 1
  batch_size: 2048

  # Predictor Configuration
  predictor_hidden: 256
  predictor_out: 1
  predictor_layers: 3
  predictor_dropout: 0.02
  k_list: [1, 3, 10]
  num_neighbors: [15, 15]
  self_loop: False
  undirected: False
  lambda_v: 0.5
  lambda_t: 0.5

# Model Configuration (GCN Example)
model:
  name: GCN
  num_layers: 3
  hidden_dim: 256
  dropout: 0.02

# Dataset Configuration (Example)
dataset:
  name: Movies
  graph_path: /home/ai/MMAG/Movies/MoviesGraph.pt
  v_emb_path: /home/ai/lys/MAG/MAGB-master/dataset/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy
  t_emb_path: /home/ai/lys/MAG/MAGB-master/dataset/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy
  lp_val_ratio: 0.2
  lp_test_ratio: 0.2
  edge_split_path: None
~~~

## Key Configuration Parameters

### Global Settings

- **seed**: Random seed for reproducibility (e.g., `42`).
- **log_dir**: Directory to save the outputs (e.g., `outputs/`).
- **device**: Device to run the model on (e.g., `cuda` for GPU).

### Link Prediction Task Settings

- **task.name**: Defines the task, set to `"lp"` for link prediction.
- **n_epochs**: The number of epochs to run the training (e.g., `1000`).
- **lr**: Learning rate for the optimizer (e.g., `0.001`).
- **fewshots**: Whether to use few-shot learning (e.g., `False`).
- **weight_decay**: Weight decay (L2 regularization) for the optimizer (e.g., `1e-5`).
- **n_runs**: Number of runs for training (e.g., `1`).
- **batch_size**: Batch size for training (e.g., `2048`).

### Predictor Configuration

- **predictor_hidden**: Hidden dimension for the link predictor (e.g., `256`).
- **predictor_out**: Output dimension for the predictor (e.g., `1`).
- **predictor_layers**: Number of layers for the predictor (e.g., `3`).
- **predictor_dropout**: Dropout rate for the predictor (e.g., `0.02`).
- **k_list**: List of top-k values for evaluation (e.g., `[1, 3, 10]`).
- **num_neighbors**: Number of neighbors to sample during training (e.g., `[15, 15]`).
- **self_loop**: Whether to add self-loops in the graph (e.g., `False`).
- **undirected**: Whether the graph is undirected (e.g., `False`).
- **lambda_v**: Weight for the first modality (e.g., `0.5`).
- **lambda_t**: Weight for the second modality (e.g., `0.5`).

### Model Configuration (GCN Example)

- **model.name**: The model to use for the task, e.g., `"GCN"`.
- **num_layers**: Number of layers in the GCN model (e.g., `3`).
- **hidden_dim**: The dimension of the hidden layers (e.g., `256`).
- **dropout**: Dropout rate to prevent overfitting (e.g., `0.02`).

### Dataset Configuration (Example)

- **dataset.name**: The name of the dataset, e.g., `"Movies"`.
- **graph_path**: Path to the graph file (e.g., `/home/ai/MMAG/Movies/MoviesGraph.pt`).
- **v_emb_path**: Path to the visual features of nodes (e.g., `/home/ai/lys/MAG/MAGB-master/dataset/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy`).
- **t_emb_path**: Path to the text features of nodes (e.g., `/home/ai/lys/MAG/MAGB-master/dataset/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy`).
- **lp_val_ratio**: Ratio of the dataset to use  for validation (e.g., `0.2`).
- **lp_test_ratio**: Ratio of the dataset to use for test (e.g., `0.2`).
- **edge_split_path**: Path for edge split (if applicable, e.g., `None`).

## Running the Link Prediction Task

### 1. Full Graph Training

To run the model on the full graph, you can import and call the function directly in your script. Here’s how you can do it:

```python
from graph_centric.lp.run import run_lp

# Create or load your configuration (cfg) using Hydra
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Print the loaded configuration for debugging purposes
    print(OmegaConf.to_yaml(cfg))

    # Call the link prediction function for full graph training
    if cfg.task.name == "lp":
        run_lp(cfg)

if __name__ == "__main__":
    main()
```

In this setup, the `run_lp` function is imported from `graph_centric.lp.run`, and then it is called within the `main` function after the configuration (`cfg`) is loaded using Hydra.

### 2. Batch-Based Subgraph Training

To train the model using subgraphs in batches, you can similarly import and call the batch-based training function. Here’s an example:

```python
from graph_centric.lp.run_batch import run_lp

# Create or load your configuration (cfg) using Hydra
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Print the loaded configuration for debugging purposes
    print(OmegaConf.to_yaml(cfg))

    # Call the link prediction function for batch-based subgraph training
    if cfg.task.name == "lp":
        run_lp(cfg)

if __name__ == "__main__":
    main()
```

In this case, `run_lp` is imported from `graph_centric.lp.run_batch` for subgraph-based training, and is invoked similarly within the `main` function.

## Hydra and Config Management

The `config.yaml` file is parsed and managed using Hydra, which automatically loads the configuration based on the given path and configuration name. The configuration is passed to the main function where it determines which task to run, which model to use, and how to configure the dataset.

Here’s a basic overview of the `main` function:

```python
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Print the configuration to ensure it has loaded correctly
    print(OmegaConf.to_yaml(cfg))

    # Determine the task and run the appropriate function
    if cfg.task.name == "lp":
        from graph_centric.lp.run_batch import run_lp  # or `run` for full graph
        run_lp(cfg)

if __name__ == "__main__":
    main()
```

- `cfg`: The configuration loaded via Hydra.

